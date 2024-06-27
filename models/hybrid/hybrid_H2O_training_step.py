# this module contains the training (also can be used for validation & test steps) step.

# load libraries
import torch
import sys

# load custom modules
import hybrid_helpers as hh
import debug_helpers as dh

# load custom modules from different directories
path_water_cycle = "../physics/water_cycle"
sys.path.insert(1, path_water_cycle) # add the module path to PATH temporarily (until this session ends)
import tws

def training_step(self, batch): # , monitor, dl_monitor, path_monitoring_results):

    """
    This function implements the training step of the hybrid ML model of full water cycle. This function is meant to be used within the training/validation/test_step of pytorch lightning.

    Arguments:
    self: A pytorch lightning model where the full forward run of the hybrid model of water cycle has been implemented.
    batch: Parameter required by pytorch lightning's training/validation/test_step
    debug: Either True or False (Boolean) indicating whether debugging code should run during training step.

    Returns: loss_sum containing the final summed loss of all the predictions
    """

    # get the features and constraints from the batch
    # forcing, static, constraints_static, constraints, _ = batch
    forcing, static, constraints, _ = batch

    # store the number of examples/grids in the current batch (needed for initialising lstm's hidden and cell states)
    batch_size = forcing.shape[0]

    # initialise the physical states with zeros
    swe0 = torch.zeros(size = (batch_size, 1), dtype = torch.float32)
    SM0 = torch.zeros(size = (batch_size, 1), dtype = torch.float32)
    GW0 = torch.zeros(size = (batch_size, 1), dtype = torch.float32)
    fAPAR_nn_0 = torch.zeros(size = (batch_size, 1), dtype = torch.float32)

    # initialise the hidden and cell states with 0's
    h0 = torch.zeros(size = (batch_size, self.hidden_size_lstm), dtype = torch.float32)
    c0 = torch.zeros(size = (batch_size, self.hidden_size_lstm), dtype = torch.float32)

    # store the state variables in a dictionary
    states0 = {
        "swe_t": swe0,
        "SM_t": SM0,
        "GW_t": GW0,
        "fAPAR_nn_t": fAPAR_nn_0,
        "h_t": h0,
        "c_t": c0
    }

    # run forward computation on spin-up mode
    with torch.no_grad(): # detach from the computational graph
        _, _, _, states_spinup = self(forcing = forcing, static = static, states_initial = states0)

    # get values & dates from the constraints
    tws_values, tws_dates = constraints["tws"]
    et_values, et_dates = constraints["et"]
    q_values, q_dates = constraints["q"]
    swe_values, swe_dates = constraints["swe"]
    fapar_values, fapar_dates = constraints["fapar"]

    # add a new dimension to make the dimension equivalent to predictions' dimensions
    tws_values = tws_values.unsqueeze(2)
    et_values = et_values.unsqueeze(2)
    q_values = q_values.unsqueeze(2)
    swe_values = swe_values.unsqueeze(2)
    fapar_values = fapar_values.unsqueeze(2)

    # make dates of constraints a long 1D tensor
    tws_dates = tws_dates[0, :]
    et_dates = et_dates[0, :]
    q_dates = q_dates[0, :]
    swe_dates = swe_dates[0, :]
    fapar_dates = fapar_dates[0, :]

    # center target tws around 0 to make it comparable to predicted tws anomalies
    tws_values = tws.compute_tws_anomaly(tws_values)

    # standardise the constraints
    tws_values_standardized = hh.standardize_single(data = tws_values, mean = torch.tensor(0.0), std = self.std_tws)
    et_values_standardized = hh.standardize_single(data = et_values, mean = self.mean_et, std = self.std_et)
    q_values_standardized = hh.standardize_single(data = q_values, mean = self.mean_q, std = self.std_q)
    # swe needs to be updated first, before standardising
    fapar_values_standardized = hh.standardize_single(data = fapar_values, mean = self.mean_fapar, std = self.std_fapar)

    # run forward hybrid computation
    preds_nn, preds_hybrid, preds_hybrid_l, _ = self(forcing = forcing, static = static, states_initial = states_spinup)

    # retain the gradients of all the physics layer
    dh.retain_grads_all(preds_hybrid_l)

    # get predicted constraint variables from the dictionary. All -> of shape (batch_size, num_time_steps, 1)
    tws_predicted = preds_hybrid["tws_anomaly_ts"]
    et_predicted = preds_hybrid["ET_ts"]
    q_predicted = preds_hybrid["runoff_total_ts"]
    swe_predicted = preds_hybrid["swe_ts"]
    fapar_predicted = preds_nn["fAPAR_nn_ts"]

    # set the target swe to prediction swe, when both targets and predictions are greater than a threshold value (default is 100)
    swe_values = hh.update_target_swe_4_loss_func(pred = swe_predicted, target = swe_values)

    # standardize the UPDATED targets of SWE
    swe_values_standardized = hh.standardize_single(data = swe_values, mean = self.mean_swe, std = self.std_swe)

    # standardize the predictions
    tws_predicted_standardized = hh.standardize_single(data = tws_predicted, mean = torch.tensor(0.0), std = self.std_tws)
    et_predicted_standardized = hh.standardize_single(data = et_predicted, mean = self.mean_et, std = self.std_et)
    q_predicted_standardized = hh.standardize_single(data = q_predicted, mean = self.mean_q, std = self.std_q)
    swe_predicted_standardized = hh.standardize_single(data = swe_predicted, mean = self.mean_swe, std = self.std_swe)
    fapar_predicted_standardized = hh.standardize_single(data = fapar_predicted, mean = self.mean_fapar, std = self.std_fapar)

    # filter the dates and values of predictions to match the dates of the corresponding target
    tws_predicted_standardized, tws_predicted_dates_filtered = hh.filter_matching_dates(data_pred = tws_predicted_standardized, dates_pred = self.dates_prediction, dates_target = tws_dates)
    et_predicted_standardized, et_predicted_dates_filtered = hh.filter_matching_dates(data_pred = et_predicted_standardized, dates_pred = self.dates_prediction, dates_target = et_dates)
    q_predicted_standardized, q_predicted_dates_filtered = hh.filter_matching_dates(data_pred = q_predicted_standardized, dates_pred = self.dates_prediction, dates_target = q_dates)
    swe_predicted_standardized, swe_predicted_dates_filtered = hh.filter_matching_dates(data_pred = swe_predicted_standardized, dates_pred = self.dates_prediction, dates_target = swe_dates)
    fapar_predicted_standardized, fapar_predicted_dates_filtered = hh.filter_matching_dates(data_pred = fapar_predicted_standardized, dates_pred = self.dates_prediction, dates_target = fapar_dates)

    # aggregate the needed targets to monthly
    tws_target_monthly = hh.aggregate_monthly_nan(tws_values_standardized, tws_dates)
    et_target_monthly = hh.aggregate_monthly_nan(et_values_standardized, et_dates)
    q_target_monthly = hh.aggregate_monthly_nan(q_values_standardized, q_dates)
    fapar_target_monthly = hh.aggregate_monthly_nan(fapar_values_standardized, fapar_dates)
    # swe is daily so no aggregation!

    # agregate the needed predictions to monthly
    tws_predicted_monthly = hh.aggregate_monthly_nan(tws_predicted_standardized, tws_predicted_dates_filtered)
    et_predicted_monthly = hh.aggregate_monthly_nan(et_predicted_standardized, et_predicted_dates_filtered)
    q_predicted_monthly = hh.aggregate_monthly_nan(q_predicted_standardized, q_predicted_dates_filtered)
    fapar_predicted_monthly = hh.aggregate_monthly_nan(fapar_predicted_standardized, fapar_predicted_dates_filtered)
    # swe is daily so no aggregation!

    # compute the loss for the predictions
    loss_tws_monthly = hh.compute_nan_mse_time(pred = tws_predicted_monthly, target = tws_target_monthly)
    loss_et_monthly = hh.compute_nan_mse_time(pred = et_predicted_monthly, target = et_target_monthly)
    loss_q_monthly = hh.compute_nan_mse_time(pred = q_predicted_monthly, target = q_target_monthly)
    loss_swe = hh.compute_nan_mse_time(pred = swe_predicted_standardized, target = swe_values_standardized)
    loss_fapar_monthly = hh.compute_nan_mse_time(pred = fapar_predicted_monthly, target = fapar_target_monthly)

    # add all the losses together
    loss_sum = loss_tws_monthly + loss_et_monthly + loss_q_monthly + loss_swe + loss_fapar_monthly # + loss_baseflow_k

    # clip the gradients of physical layers
    dh.clip_physics_gradients(preds_hybrid_l, clip_value = torch.tensor(1000000.))

    # store all the individual losses (inc. sum) together in a dictionary
    losses_all = {"tws": loss_tws_monthly,
                  "et": loss_et_monthly,
                  "runoff": loss_q_monthly,
                  "swe": loss_swe,
                  "fapar": loss_fapar_monthly,
                  "loss_sum": loss_sum}

    # return the final loss
    return losses_all