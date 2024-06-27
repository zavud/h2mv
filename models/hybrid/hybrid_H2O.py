# This module contains the script for the hybrid model of water cycle
# load libraries
import sys
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

# load the custom modules
import hybrid_helpers as hh
import hybrid_H2O_training_step
import hybrid_H2O_common_step
import debug_helpers


# import modules that are not stored in the current directory
path_water_cycle = "../physics/water_cycle"
path_ZarrDataset = "../../datasets"
path_nn = "../neural_networks"
sys.path.insert(1, path_water_cycle) # add the module path to PATH temporarily (until this session ends)
sys.path.insert(2, path_ZarrDataset)
sys.path.insert(3, path_nn)
import tws # needed for computing anomalies of tws
import water_cycle_forward
import ZarrDataset as zd
import neural_networks

# define the LightningModule
class HybridH2O(pl.LightningModule): # inherits from LightningModule

    # define the init method
    def __init__(self, zarr_data_path, device, k): # , monitor, path_monitoring_results):

        super().__init__() # initialize the parent class

        # store the device within init
        self.device2 = device # adding ..2 to the name is needed since ligthning modules also have a device parameter

        # store the k'th fold for cross validation
        self.k = k

        # store the data path to the zarr file containing training/val/test sets
        self.zarr_data_path = zarr_data_path

        # automatically save all the hyperparameters passed to init
        self.save_hyperparameters()

        # instantiate ZarrDataset class to collect statistical info
        self.dataset = zd.ZarrDataset(zarr_data_path = self.zarr_data_path, data_split = "training", k = self.k)

        # get dates for predictions
        self.dates_prediction = hh.get_dates_as_tensor(self.dataset.forcing_xr[0]).to(self.device2) # num_time_steps in any forcing must be equal to num_time_steps in predictions

        # get the mean and std of the features (needed for normalising)
        self.means_forcing, self.means_static = hh.extract_statistics_features(dataset = self.dataset, stat = "mean")
        self.stds_forcing, self.stds_static = hh.extract_statistics_features(dataset = self.dataset, stat = "std")
        
        # send them to the given device
        self.means_forcing = self.means_forcing.to(self.device2)
        self.means_static = self.means_static.to(self.device2)
        self.stds_forcing = self.stds_forcing.to(self.device2)
        self.stds_static = self.stds_static.to(self.device2)

        # get the mean for constraints (needed for normalising)
        self.mean_tws = hh.extract_statistics_single(dataset = self.dataset.constraints_xr[0], stat = "mean").to(self.device2)
        self.mean_et = hh.extract_statistics_single(dataset = self.dataset.constraints_xr[1], stat = "mean").to(self.device2)
        self.mean_q = hh.extract_statistics_single(dataset = self.dataset.constraints_xr[2], stat = "mean").to(self.device2)
        self.mean_swe = hh.extract_statistics_single(dataset = self.dataset.constraints_xr[3], stat = "mean").to(self.device2)
        self.mean_fapar = hh.extract_statistics_single(dataset = self.dataset.constraints_xr[4], stat = "mean").to(self.device2)

        # get the standard deviation for constraints (needed for normalising)
        self.std_tws = hh.extract_statistics_single(dataset = self.dataset.constraints_xr[0], stat = "std").to(self.device2)
        self.std_et = hh.extract_statistics_single(dataset = self.dataset.constraints_xr[1], stat = "std").to(self.device2)
        self.std_q = hh.extract_statistics_single(dataset = self.dataset.constraints_xr[2], stat = "std").to(self.device2)
        self.std_swe = hh.extract_statistics_single(dataset = self.dataset.constraints_xr[3], stat = "std").to(self.device2)
        self.std_fapar = hh.extract_statistics_single(dataset = self.dataset.constraints_xr[4], stat = "std").to(self.device2)

        # store the dimension info
        num_forcing = 3
        num_static = 30
        num_static_condensed = 12
        num_states = 4
        num_features = num_forcing + num_static_condensed + num_states

        # store the dimensions of needed NN parameters
        self.hidden_size_lstm = 100

        # initialise the global constant variables (fixed in time and space - 0D learnable weights)
        self.beta_snow = nn.Parameter(torch.tensor(3.0, dtype = torch.float32))
        self.beta_baseflow = nn.Parameter(torch.tensor(-6, dtype = torch.float32))

        # Fully connected NN module that takes the static inputs and gives condensed representation of it as well as static predictions
        self.fc_static = neural_networks.FC_Static(num_static_input = num_static, num_static_out = 1, hidden_size1 = 150, hidden_size2 = 12)

        # LSTM layer that takes meteorological forcing (time-series) as well as condensed repr. of static outputs and gives temporal intermediate NN preds.
        self.lstm_fc = neural_networks.LSTM_Dynamic(num_features = num_features, num_temporal_outputs = 7, hidden_size_lstm = self.hidden_size_lstm)

    # define the hybrid forward computation
    def forward(self, forcing: torch.Tensor, static: torch.Tensor, states_initial: dict, prediction_mode = False):

        # activate global constants (beta parameters)
        beta_snow = torch.sigmoid(self.beta_snow)
        beta_baseflow = torch.sigmoid(self.beta_baseflow)

        # create empty lists that will store the direct NN predictions at each time step
        alpha_snow_melt_ts = []
        alpha_Ei_ts = []
        alpha_Es_ts = []
        alpha_T_ts = []
        alpha_r_soil_ts = []
        alpha_r_gw_ts = []
        fAPAR_nn_ts = []

        # create empty lists that will store the WATER CYCLE components at each time step
        snow_acc_ts = []
        snow_melt_ts = []
        swe_ts = []
        Ei_ts = []
        Es_ts = []
        T_ts = []
        ET_ts = []
        EF_ts = []
        water_input_ts = []
        r_soil_t_fractions = []
        r_soil_ts = []
        r_gw_fraction_ts = []
        r_gw_ts = []
        SM_ts = []
        rel_SM_ts = []
        runoff_surface_fraction_ts = []
        runoff_surface_ts = []
        baseflow_ts = []
        runoff_total_ts = []
        GW_ts = []
        tws_ts = []
        prec_actual_ts = []

        # store the number of time steps (or sequences) (needed for looping through each time)
        num_time_steps = forcing.shape[1]

        # get the initial physical states from states_initial
        swe_t = states_initial["swe_t"].to(self.device2)
        SM_t = states_initial["SM_t"].to(self.device2)
        GW_t = states_initial["GW_t"].to(self.device2)
        fAPAR_nn_t = states_initial["fAPAR_nn_t"].to(self.device2)

        # get LSTM's initial cell and hidden states from states_initial
        h_t = states_initial["h_t"].to(self.device2)
        c_t = states_initial["c_t"].to(self.device2)

        # normalise forcing
        static_normalised = hh.standardize_input(data = static, means = self.means_static, stds = self.stds_static)

        # forward computation to get static outputs & condensed representation of static inputs
        static_condensed, static_output = self.fc_static(static_normalised) # of shape (batch_size, hidden_size2) & (batch_size, num_static_out)

        # extract static outputs separately
        sm_max = static_output[:, 0].unsqueeze(1) + torch.tensor(100., dtype = torch.float32) # adding 100 helps the model to predict higher values

        # activate static preds
        sm_max = F.softplus(sm_max) + torch.tensor(1.0, dtype = torch.float32) # sm_max is not allowed to be smaller than 1. Of shape (batch_size, 1)

        # forward computation to predict alpha and fAPAR (time varying) variables (forcing + static -> lstm -> fc -> alpha outputs + fAPAR)
        for t in range(num_time_steps): # loop through each time step

            # 'roughly-normalise' states to give as an input
            swe_t_normalised = swe_t / 100
            SM_t_normalised = SM_t / 200
            GW_t_normalised = GW_t / 200
            # fapar does not need normalisation as its bounded between 0-1

            # select features at the current time step t
            forcing_t = forcing[:, t, :] # of shape (batch_size, num_forcing_variables + num_static_variables)

            # extract forcing variables separately (needed for physical computations)
            rn_t = forcing_t[:, 0].unsqueeze(1) # of shape (batch_size, 1)
            prec_t = forcing_t[:, 1].unsqueeze(1) # of shape (batch_size, 1)
            tair_t = forcing_t[:, 2].unsqueeze(1) # of shape (batch_size, 1)

            # normalise forcing
            forcing_t_normalised = hh.standardize_input(data = forcing_t, means = self.means_forcing, stds = self.stds_forcing)

            # combine features and normalised states
            features_t = torch.cat((forcing_t_normalised, static_condensed, swe_t_normalised, SM_t_normalised, GW_t_normalised, fAPAR_nn_t), 1)

            # forward computation to get LSTM states as well as temporal predictions
            h_t, c_t, output_t = self.lstm_fc(features_t, h_t, c_t) # of shapes (batch_size, hidden_size_lstm), (batch_size, hidden_size_lstm), (batch_size, num_temporal_outputs)

            # extract the NN learned alpha parameters and fAPAR for using in the physical part of the model
            alpha_snow_melt_t = output_t[:, 0].unsqueeze(1) # of shape (batch_size, 1)
            alpha_Ei_t = output_t[:, 1].unsqueeze(1) # of shape (batch_size, 1)
            alpha_Es_t = output_t[:, 2].unsqueeze(1) # of shape (batch_size, 1)
            alpha_T_t = output_t[:, 3].unsqueeze(1) # of shape (batch_size, 1)
            alpha_r_soil_t = output_t[:, 4].unsqueeze(1) # of shape (batch_size, 1)
            alpha_r_gw_t = output_t[:, 5].unsqueeze(1) # of shape (batch_size, 1)
            fAPAR_nn_t = output_t[:, 6].unsqueeze(1) # of shape (batch_size, 1)

            # apply activation functions on the time-varied outputs
            alpha_snow_melt_t = F.softplus(alpha_snow_melt_t) # of shape (batch_size, 1)
            alpha_Ei_t = F.softplus(alpha_Ei_t) # of shape (batch_size, 1)
            alpha_Es_t = torch.sigmoid(alpha_Es_t) # of shape (batch_size, 1)
            alpha_T_t = torch.sigmoid(alpha_T_t) # of shape (batch_size, 1)
            alpha_r_soil_t = torch.sigmoid(alpha_r_soil_t) # of shape (batch_size, 1)
            alpha_r_gw_t = torch.sigmoid(alpha_r_gw_t) # of shape (batch_size, 1)
            fAPAR_nn_t = torch.sigmoid(fAPAR_nn_t) # of shape (batch_size, 1)

            ### Forward run of water cycle
            
            # prepare the input to the forward run of water cycle
            water_cycle_input = {
                "tair_t": tair_t,
                "prec_t": prec_t,
                "rn_t": rn_t,
                "swe_t_prev": swe_t,
                "SM_t_prev": SM_t,
                "GW_t_prev": GW_t,
                "alpha_snow_melt_t": alpha_snow_melt_t,
                "alpha_Ei_t": alpha_Ei_t,
                "alpha_Es_t": alpha_Es_t,
                "alpha_T_t": alpha_T_t,
                "alpha_r_soil_t": alpha_r_soil_t,
                "alpha_r_gw_t": alpha_r_gw_t,
                "fAPAR_nn_t": fAPAR_nn_t,
                "SM_max_nn": sm_max,
                "beta_snow": beta_snow,
                "beta_baseflow": beta_baseflow
            }

            # run full forward computation of the water cycle
            water_cycle_output = water_cycle_forward.run_water_cycle_forward(water_cycle_input = water_cycle_input)

            if prediction_mode:

                # extract the outputs from the forward run of water cycle
                snow_acc_t = water_cycle_output["snow_acc_t"]
                snow_melt_t = water_cycle_output["snow_melt_t"]
                swe_t = water_cycle_output["swe_t"]
                Ei_t = water_cycle_output["Ei_t"]
                Es_t = water_cycle_output["Es_t"]
                T_t = water_cycle_output["T_t"]
                ET_t = water_cycle_output["ET_t"]
                EF_t = water_cycle_output["EF_t"]
                water_input_t = water_cycle_output["water_input_t"]
                r_soil_t_fraction = water_cycle_output["r_soil_t_fraction"]
                r_soil_t = water_cycle_output["r_soil_t"]
                r_gw_fraction_t = water_cycle_output["r_gw_fraction_t"]
                r_gw_t = water_cycle_output["r_gw_t"]
                SM_t = water_cycle_output["SM_t"]
                rel_SM_t = water_cycle_output["rel_SM_t"]
                runoff_surface_fraction_t = water_cycle_output["runoff_surface_fraction_t"]
                runoff_surface_t = water_cycle_output["runoff_surface_t"]
                baseflow_t = water_cycle_output["baseflow_t"]
                runoff_total_t = water_cycle_output["runoff_total_t"]
                GW_t = water_cycle_output["GW_t"]
                tws_t = water_cycle_output["tws_t"]
                prec_actual_t = water_cycle_output["prec_actual"]

                # append the direct NN predictions for all the times
                alpha_snow_melt_ts.append(alpha_snow_melt_t)
                alpha_Ei_ts.append(alpha_Ei_t)
                alpha_Es_ts.append(alpha_Es_t)
                alpha_T_ts.append(alpha_T_t)
                alpha_r_soil_ts.append(alpha_r_soil_t)
                alpha_r_gw_ts.append(alpha_r_gw_t)
                fAPAR_nn_ts.append(fAPAR_nn_t)

                # append the WATER CYCLE components for all the times
                snow_acc_ts.append(snow_acc_t)
                snow_melt_ts.append(snow_melt_t)
                swe_ts.append(swe_t)
                Ei_ts.append(Ei_t)
                Es_ts.append(Es_t)
                T_ts.append(T_t)
                ET_ts.append(ET_t)
                EF_ts.append(EF_t)
                water_input_ts.append(water_input_t)
                r_soil_t_fractions.append(r_soil_t_fraction)
                r_soil_ts.append(r_soil_t)
                r_gw_fraction_ts.append(r_gw_fraction_t)
                r_gw_ts.append(r_gw_t)
                SM_ts.append(SM_t)
                rel_SM_ts.append(rel_SM_t)
                runoff_surface_fraction_ts.append(runoff_surface_fraction_t)
                runoff_surface_ts.append(runoff_surface_t)
                baseflow_ts.append(baseflow_t)
                runoff_total_ts.append(runoff_total_t)
                GW_ts.append(GW_t)
                tws_ts.append(tws_t)
                prec_actual_ts.append(prec_actual_t)
            
            else: # training mode

                # extract the outputs from the forward run of water cycle
                # constraints
                swe_t = water_cycle_output["swe_t"]
                ET_t = water_cycle_output["ET_t"]
                tws_t = water_cycle_output["tws_t"]
                runoff_total_t = water_cycle_output["runoff_total_t"]

                # unconstrained states (needed for input to LSTM in the next step)
                SM_t = water_cycle_output["SM_t"]
                GW_t = water_cycle_output["GW_t"]

                # append the direct NN predictions for all the times
                fAPAR_nn_ts.append(fAPAR_nn_t)

                # append the WATER CYCLE components for all the times
                swe_ts.append(swe_t)
                ET_ts.append(ET_t)
                tws_ts.append(tws_t)
                runoff_total_ts.append(runoff_total_t)

        if prediction_mode:

            # get anomalies of tws
            tws_ts_tensor = torch.stack(tws_ts, dim = 1) # of shape (batch_size, num_time_steps, 1)
            tws_anomaly_ts = tws.compute_tws_anomaly(tws = tws_ts_tensor) # of shape (batch_size, num_time_steps, 1) & (batch_size, 1, 1)

            # store the learned global constant variables in a dictionary
            # of shape (0D)
            learned_constants = {
                "beta_snow": beta_snow,
                "beta_baseflow": beta_baseflow
            }

            # store the static outputs in a dictionary
            # of shape (batch_size, 1)
            preds_static = {
                "sm_max": sm_max
            }

            # convert the list of tensors (containing direct NN predictions) into a dictionary containing single 3D arrays
            # of shape (batch_size, num_time_steps, 1)
            preds_nn = {
                "alpha_snow_melt_ts": torch.stack(alpha_snow_melt_ts, dim = 1),
                "alpha_Ei_ts": torch.stack(alpha_Ei_ts, dim = 1),
                "alpha_Es_ts": torch.stack(alpha_Es_ts, dim = 1),
                "alpha_T_ts": torch.stack(alpha_T_ts, dim = 1),
                "alpha_r_soil_ts": torch.stack(alpha_r_soil_ts, dim = 1),
                "alpha_r_gw_ts": torch.stack(alpha_r_gw_ts, dim = 1),
                "fAPAR_nn_ts": torch.stack(fAPAR_nn_ts, dim = 1)
            }

            # convert the list of tensors (containing the hybrid predictions) into a dictionary single 3D arrays
            # of shape (batch_size, num_time_steps, 1)
            preds_hybrid = {
                "snow_acc_ts": torch.stack(snow_acc_ts, dim = 1),
                "snow_melt_ts": torch.stack(snow_melt_ts, dim = 1),
                "swe_ts": torch.stack(swe_ts, dim = 1),
                "Ei_ts": torch.stack(Ei_ts, dim = 1),
                "Es_ts": torch.stack(Es_ts, dim = 1),
                "T_ts": torch.stack(T_ts, dim = 1),
                "ET_ts": torch.stack(ET_ts, dim = 1),
                "EF_ts": torch.stack(EF_ts, dim = 1),
                "water_input_ts": torch.stack(water_input_ts, dim = 1),
                "r_soil_t_fractions": torch.stack(r_soil_t_fractions, dim = 1),
                "r_soil_ts": torch.stack(r_soil_ts, dim = 1),
                "r_gw_fraction_ts": torch.stack(r_gw_fraction_ts, dim = 1),
                "r_gw_ts": torch.stack(r_gw_ts, dim = 1),
                "SM_ts": torch.stack(SM_ts, dim = 1),
                "rel_SM_ts": torch.stack(rel_SM_ts, dim = 1),
                "runoff_surface_fraction_ts": torch.stack(runoff_surface_fraction_ts, dim = 1),
                "runoff_surface_ts": torch.stack(runoff_surface_ts, dim = 1),
                "baseflow_ts": torch.stack(baseflow_ts, dim = 1),
                "runoff_total_ts": torch.stack(runoff_total_ts, dim = 1),
                "GW_ts": torch.stack(GW_ts, dim = 1),
                "tws_ts": tws_ts_tensor,
                "tws_anomaly_ts": tws_anomaly_ts,
                "prec_actual_ts": torch.stack(prec_actual_ts, dim = 1)
            }

            # return the final dictionaries
            return learned_constants, preds_static, static_condensed, preds_nn, preds_hybrid
        
        else: # training mode

            # get anomalies of tws
            tws_ts_tensor = torch.stack(tws_ts, dim = 1) # of shape (batch_size, num_time_steps, 1)
            tws_anomaly_ts = tws.compute_tws_anomaly(tws = tws_ts_tensor) # of shape (batch_size, num_time_steps, 1) & (batch_size, 1, 1)

            # for debugging, tensors of lists are needed
            preds_hybrid_l = {
                # "snow_acc_ts": snow_acc_ts,
                # "snow_melt_ts": snow_melt_ts,
                # "swe_ts": swe_ts,
                # "Ei_ts": Ei_ts,
                # "Es_ts": Es_ts,
                # "T_ts": T_ts,
                # "water_input_ts": water_input_ts,
                # "ET_ts": ET_ts,
                # "r_soil_t_fractions": r_soil_t_fractions,
                # "r_soil_ts": r_soil_ts,
                # "r_gw_ts": r_gw_ts,
                "SM_ts": SM_ts,
                # "runoff_surface_ts": runoff_surface_ts,
                # "baseflow_ts": baseflow_ts,
                # "runoff_total_ts": runoff_total_ts,
                # "GW_ts": GW_ts,
                # "tws_ts": tws_ts
            }

            # convert the list of tensors (containing direct NN predictions) into a dictionary containing single 3D arrays
            # of shape (batch_size, num_time_steps, 1)
            preds_nn = {
                "fAPAR_nn_ts": torch.stack(fAPAR_nn_ts, dim = 1)
            }

            # convert the list of tensors (containing the hybrid predictions) into a dictionary single 3D arrays
            # of shape (batch_size, num_time_steps, 1)
            preds_hybrid = {
                "swe_ts": torch.stack(swe_ts, dim = 1),
                "ET_ts": torch.stack(ET_ts, dim = 1),
                "runoff_total_ts": torch.stack(runoff_total_ts, dim = 1),
                "tws_anomaly_ts": tws_anomaly_ts,
            }

            # make a dictionary containing the final states (physical & LSTM)
            states_final = {
                "swe_t": swe_t,
                "SM_t": SM_t,
                "GW_t": GW_t,
                "fAPAR_nn_t": fAPAR_nn_t,
                "h_t": h_t,
                "c_t": c_t
            }

            # return the final dictionaries
            return preds_nn, preds_hybrid, preds_hybrid_l, states_final
    
    # define the training loop/step
    def training_step(self, batch, batch_idx):

        # run the training step
        losses_all = hybrid_H2O_training_step.training_step(self = self, batch = batch)

        # log the losses for each epoch
        for loss_name, loss in losses_all.items():

            if loss_name == "loss_sum": # show sum of the losses in the progress bar 

                self.log(loss_name + "_training", loss, on_step = False, on_epoch = True, prog_bar = True)
            
            else:

                self.log(loss_name + "_training", loss, on_step = False, on_epoch = True, prog_bar = False)

        # if the torch is a finite value
        if torch.isfinite(losses_all["loss_sum"]):

            # return the final loss
            return losses_all["loss_sum"]
        
        # the edge cases where the loss is not a finite number, e.g. due to instable or issues with samples in the batch
        else: 

            # print a message and show the current losses when this happens
            print(f'Skipping to update gradients for this batch, because the total loss is {losses_all["loss_sum"]}, loss_tws_monthly: {losses_all["loss_tws_monthly"]}, loss_et_monthly: {losses_all["loss_et_monthly"]}, loss_q_monthly: {losses_all["loss_q_monthly"]}, loss_swe, {losses_all["loss_swe"]}, loss_fapar_monthly:  {losses_all["loss_fapar_monthly"]}')

            return None # source: https://github.com/Lightning-AI/pytorch-lightning/issues/4956#issuecomment-738345169
    
    # define the validation step
    def validation_step(self, batch, batch_idx):

        # run the training step
        losses_all = hybrid_H2O_common_step.common_step(self = self, batch = batch)
        
        # log the losses for each epoch
        for loss_name, loss in losses_all.items():

            if loss_name == "loss_sum": # show sum of the losses in the progress bar 

                self.log(loss_name + "_validation", loss, on_step = False, on_epoch = True, prog_bar = True)
            
            else:

                self.log(loss_name + "_validation", loss, on_step = False, on_epoch = True, prog_bar = False)

        # return the loss for swe
        return losses_all["loss_sum"]

    # define the validation step
    def test_step(self, batch, batch_idx):

        # run the training step
        losses_all = hybrid_H2O_common_step.common_step(self = self, batch = batch)
        
        # log the losses for each epoch
        for loss_name, loss in losses_all.items():

            self.log(loss_name + "_testing", loss, on_step = False, on_epoch = True)

        # return the loss for swe
        return losses_all["loss_sum"]
    
    # define the optimizer for training the model
    def configure_optimizers(self):
        
        # choose Adam for now
        optimizer = torch.optim.AdamW(self.parameters(), lr = 0.01)

        return optimizer
    
    # define train dataloader
    def train_dataloader(self, data_split = "training"):

        # instantiate the training dataset
        dataset = zd.ZarrDataset(zarr_data_path = self.zarr_data_path, data_split = data_split, k = self.k)

        # define dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = 128 * 2, shuffle = True, num_workers = 10)

        # return the dataloader
        return dataloader

    # define validation dataloader
    def val_dataloader(self, data_split = "validation"):

        # instantiate the training dataset
        val_dataset = zd.ZarrDataset(zarr_data_path = self.zarr_data_path, data_split = data_split, k = self.k)

        # define dataloader
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = 128 * 2, num_workers = 10)

        # return the dataloader
        return val_dataloader

    # define test dataloader
    def test_dataloader(self, data_split = "testing"):

        # instantiate the training dataset
        test_dataset = zd.ZarrDataset(zarr_data_path = self.zarr_data_path, data_split = data_split, k = self.k)

        # define dataloader
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 128 * 2, num_workers = 10)

        # return the dataloader
        return test_dataloader
    
    # define global dataloader
    def global_dataloader(self, data_split = "global"):

        # instantiate the training dataset
        global_dataset = zd.ZarrDataset(zarr_data_path = self.zarr_data_path, data_split = data_split, k = self.k)

        # define dataloader
        global_dataloader = torch.utils.data.DataLoader(global_dataset, batch_size = 128 * 2, num_workers = 10)

        # return the dataloader
        return global_dataloader
    
    # define prediction step
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
    
        # get the features and constraints from the batch
        # forcing, static, constraints_static, constraints_temporal, coords = batch
        forcing, static, constraints_temporal, coords = batch

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
        _, _, _, states_spinup = self(forcing = forcing, static = static, states_initial = states0)

        # run forward hybrid computation
        learned_constants, preds_static, static_condensed, preds_nn, preds_hybrid = self(forcing = forcing, static = static, states_initial = states_spinup, prediction_mode = True)

        # store the output from the forward run in a tuple
        outputs = (learned_constants, preds_static, preds_nn, preds_hybrid)

        # store the constraints and corresponding coordinates in a tuple
        constraints_coords = (constraints_temporal, coords)

        # return the needed variables
        return constraints_coords, outputs