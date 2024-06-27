# This module contains helper functions used within hybrid model forward computations

# load libraries
import sys
import torch
import xarray as xr
import numpy as np

# store the path to the forcing2swe directory
module_path = "../../datasets"

# add the module path to PATH temporarily (until this session ends)
sys.path.insert(1, module_path)

# load the custom ZarrDataset module
import ZarrDataset as zd

# extract statistical data from pytorch's dataset
def extract_statistics_features(dataset: zd.ZarrDataset, stat: str):

    """
    This function extracts statistical data (either mean or standard deviation) for the forcing variables and static data 
     from the custom dataset ZarrDataset that inherits from pytorch's dataset.

    Arguments:
    dataset: Custom dataset ZarrDataset containing information about the input data
    stat: The name of the statistical variable - either "mean" or "std"

    Returns: stats containing the data of the chosen statistical variable. Torch tensor of shape (1, num_forcing_variables + num_static_variables)
    """

    # store statistical variable's names
    stat_names = ["mean", "std"]

    # check whether the types of the given arguments are correct
    # assert isinstance(dataset, zd.ZarrDataset), "dataset must be a type of ZarrDataset.ZarrDataset (zd.ZarrDataset)"
    # assert isinstance(stat, str), "stat must be a string"
    assert stat in stat_names, f"stat must be either {stat_names[0]} or {stat_names[1]}"

    # extract statistical data for each forcing variable from xarray's attributes and store it in a list
    stat_forcing = [forcing_data.attrs[stat] for forcing_data in dataset.forcing_xr]

    # extract statistical data for each static variable from xarray's attributes and store it in a list
    stat_static = dataset.static_xr.attrs[stat]

    # convert both lists into torch tensors
    stat_forcing = torch.tensor(stat_forcing, dtype = torch.float32)[None, :] # of shape (1, num_forcing_variables)
    stat_static = torch.tensor(stat_static, dtype = torch.float32)[None, :] # of shape (1, num_static_variables)

    # return the final array
    return stat_forcing, stat_static

def extract_statistics_single(dataset: xr.core.dataset.Dataset, stat: str):

    """
    This function quickly extracts the statistical information from a single dataset's attributes.

    Arguments:
    dataset: An xarray Dataset
    stat: The name of the statistical variable - either "mean" or "std"

    Returns: stat_tensor containing the data of the chosen statistical variable. Torch tensor of shape (0D)
    """

    # store statistical variable's names
    stat_names = ["mean", "std"]

    # check whether the types of the given arguments are correct
    # assert isinstance(dataset, xr.core.dataset.Dataset), "dataset must be a type of xr.core.dataset.Dataset"
    # assert isinstance(stat, str), "stat must be a string"
    assert stat in stat_names, f"stat must be either {stat_names[0]} or {stat_names[1]}"

    # extract the statistical value
    stat = dataset.attrs[stat]

    # convert stat to tensor with matching dimensions of constraint data
    stat_tensor = torch.tensor(stat, dtype = torch.float32) # of shape (0D)

    # return the final tensor
    return stat_tensor


# standardize input data
def standardize_input(data: torch.Tensor, means: torch.Tensor, stds: torch.Tensor):

    """
    This function applies the classical Z transformation on the input data.

    Arguments:
    data: input data to be standardized. Torch tensor of shape (batch_size, num_time_steps, num_forcing_variables + num_static_variables)
    means: mean values of the corresponding variables in the corresponding input data. Torch tensor of shape (1, 1, num_forcing_variables + num_static_variables)
    stds: standard deviation values of the corresponding variables in the corresponding input data. Torch tensor of shape (1, 1, num_forcing_variables + num_static_variables)

    Returns: normalized containing the normalized version of the input data. Torch tensor of shape (batch_size, num_time_steps, num_forcing_variables + num_static_variables)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(data, torch.Tensor), "data must be a type of torch.Tensor"
    # assert isinstance(means, torch.Tensor), "means must be a type of torch.Tensor"
    # assert isinstance(stds, torch.Tensor), "stds must be a type of torch.Tensor"

    # apply Z transformation
    normalized = (data - means) / stds # of shape (batch_size, num_time_steps, num_forcing_variables + num_static_variables)

    # return the normalized data
    return normalized

# standardize single
def standardize_single(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):

    """
    This function applies the classical Z transformation on a single data.

    Arguments:
    data: single data to be standardized. Torch tensor of shape (batch_size, num_time_steps, 1)
    means: mean value of given data. Torch tensor of shape (0D)
    stds: standard deviation of given data. Torch tensor of shape (0D)

    Returns: normalized containing the normalized version of the given data. Torch tensor of shape (0D)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(data, torch.Tensor), "data must be a type of torch.Tensor"
    # assert isinstance(mean, torch.Tensor), "means must be a type of torch.Tensor"
    # assert isinstance(std, torch.Tensor), "stds must be a type of torch.Tensor"

    # apply Z transformation
    normalized = (data - mean) / std # of shape (batch_size, num_time_steps, 1)

    # return the normalized data
    return normalized

# update target baseflow_k for (only) loss function
def update_target_baseflow_k_4_loss_func(pred: torch.Tensor, target: torch.Tensor):

    """
    This function sets targets with lots of snow (stored as nan in targets) to predictions. This step is done due to an issue
     in how the baseflow K parameter was measured in the original study.
    
    Arguments:
    pred: predictions that are varied in space. Torch tensor of shape (batch_size, 1)
    target: targets that are varied in time & space. Torch tensor of shape (batch_size, 1)

    Returns: 
    target: Updated targets that are varied in space. Torch tensor of shape (batch_size, 1)
    """

    # if baseflow_k is nan - meaning, there is lots of snow - we set predictions to target so that loss becomes 0
    target = torch.where(target.isnan(), pred, target)

    # return the updated target
    return target

# update target snow for (only) loss function
def update_target_swe_4_loss_func(pred: torch.Tensor, target: torch.Tensor, threshold = 100):

    """
    This function sets targets to predictions only when the corresponding targets and predictions are greater than a threshold (100 is default).
     This is done due to GLOBSNOW product being saturated when snow is high. This function makes sure that, if the target and predictions are high enough,
     the loss will be 0, and the higher predictions of snow will not be penalised.

    Arguments:
    pred: predictions that are varied in time & space. Torch tensor of shape (batch_size, num_time_steps, 1)
    target: targets that are varied in time & space. Torch tensor of shape (batch_size, num_time_steps, 1)

    Returns: 
    target: Updated targets that are varied in time & space. Torch tensor of shape (batch_size, num_time_steps, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(pred, torch.Tensor), "pred must be a type of torch.Tensor"
    # assert isinstance(target, torch.Tensor), "target must be a type of torch.Tensor"

    # conditionally make a mask when the corresponding targets and predictions are greater than a threshold value (e.g. 100)
    mask = (target > threshold) & (pred > threshold) # this is a boolean (True's if the condition is met, otherwise False) with the shape of (batch_size, num_time_steps, 1)

    # set the values of targets to predictions where mask is True
    target[mask] = pred[mask]

    # return the updated target
    return target


# custom nan_mse_time loss function
def compute_nan_mse_time(pred: torch.Tensor, target: torch.Tensor):

    """
    This function computes mean squared error between the time varying predictions and targets. It deals with the missing values in the target.

    Arguments:
    pred: predictions that are varied in time & space. Torch tensor of shape (batch_size, num_time_steps, 1)
    target: targets that are varied in time & space. Torch tensor of shape (batch_size, num_time_steps, 1)

    Returns: mse_batch containing the mean squared error for the batch (scaler)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(pred, torch.Tensor), "pred must be a type of torch.Tensor"
    # assert isinstance(target, torch.Tensor), "target must be a type of torch.Tensor"

    # find values in the sequence of target that are not missing (not nan)
    is_finite = torch.isfinite(target) # of shape (batch_size, num_time_steps, 1)

    # set the missing values in the sequences of target to the corresponding predictions
    # this is equivalent to having loss = 0 for these time steps as 'pred - pred = 0'
    target = torch.where(is_finite, target, pred) # of shape (batch_size, num_time_steps, 1)

    # compute the error for each time step
    e_t = target - pred # of shape (batch_size, num_time_steps, 1)

    # compute squared error for each time step
    se_t = torch.square(e_t) # of shape (batch_size, num_time_steps, 1)

    # compute squared error for each grid in the batch
    se_batch_grid = se_t.sum(dim = (1, 2)) # of shape (batch_size)

    # find the number of time steps (for each grid in the batch) that are not missing
    num_finite_time_steps = is_finite.sum(dim = (1, 2)) # of shape (batch_size)

    # compute mean squared error for each grid in the batch
    mse_batch_grid = se_batch_grid / num_finite_time_steps # of shape (batch_size)

    # compute mean squared error for the current batch
    mse_batch = mse_batch_grid.mean() # of shape (0D = scaler)

    # return the loss for the current batch
    return mse_batch

# function for aggregating time-series data into monthly average values
def aggregate_monthly_nan(values: torch.Tensor, dates: torch.Tensor):

    """
    This function takes values and dates of the values as inputs and computes mean value (ignoring nan values) for each month.

    Arguments:
    values: Torch tensor of shape (batch_size, num_time_steps, 1)
    dates: Dates corresponding to the values in the num_time_steps dimension. Torch tensor of shape 1D (long tensor)

    Returns: monthly_mean_tensor containing average value for each month in the values. Torch tensor of shape (batch_size, num_available_months, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(values, torch.Tensor), "values must be a type of torch.Tensor"
    # assert isinstance(dates, torch.Tensor), "dates must be a type of torch.Tensor"

    # find the number of unique dates
    _, unique_date_counts = torch.unique(dates, return_counts = True)

    # compute the cumulative sum of unique dates (e.g. how many times each unique date occur)
    indices = torch.cumsum(unique_date_counts, dim = 0)[:-1] # of shape (1D long tensor)

    # move indices to cpu otherwise torch.tensor_split gives error
    # copied from pytorch website: If indices_or_sections is a tensor, it must be a zero-dimensional or one-dimensional long tensor on the CPU
    indices = indices.to("cpu")

    # split the input tensor (values) into n number of tensors where n is the number of unique dates
    splits = torch.tensor_split(values, indices, dim = 1) # a tuple containing tensors for each month. Each tensor has shape of (batch_size, num_available_dates_in_the_month, 1)

    # compute mean value for each month in split
    monthly_mean_list = [month.nanmean(dim = 1, keepdim = True) for month in splits] # a list of tensors with average value for each month. Each tensor has a shape of (batch_size, 1, 1) 

    # concatenate/combine all the separate monthly mean along dim = 1
    monthly_mean_tensor = torch.cat(monthly_mean_list, dim = 1) # of shape (batch_size, num_available_months, 1)

    # return the monthly mean data
    return monthly_mean_tensor

# get dates from xarray dataset
def get_dates_as_tensor(dataset: xr.core.dataset.Dataset):

    """
    This function takes a dataset as input and returns the actual dates with each month having a unique 'code' (precisely: number of months since the date 1970-01).

    Arguments:
    dataset: An xarray dataset

    Returns: dates containing unique code for each month. Torch tensor of shape (1D) long tensor
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(dataset, xr.core.dataset.Dataset), "dataset must be a type of xr.core.dataset.Dataset"

    # get the dates from the input dataset
    dates = torch.from_numpy(dataset.time.values.astype("datetime64[M]").astype(np.int64))

    # return dates
    return dates

# filter only matching dates between predictions and targets
def filter_matching_dates(data_pred: torch.Tensor, dates_pred: torch.Tensor, dates_target: torch.Tensor):

    """
    This function compares prediction dates to target dates and returns the dates and values of prediction that also exist in target.

    Arguments:
    data_pred: Predicted values: Torch tensor of shape (batch_size, num_time_steps_of_predictions, 1)
    dates_pred: Dates of predictions. Torch tensor of shape 1D (length of num_time_steps_of_predictions)
    dates_target: Dates of target. Torch tensor of shape 1D (length of num_time_steps_of_target)

    Returns: A tuple (data_pred_filtered, dates_pred_filtered) containing filtered predictions and dates based on target dates. 
        data_pred_filtered is a torch tensor of shape (batch_size, num_time_steps_of_predictions_matching_target_dates, 1)
        dates_pred_filtered is a torch tensor of shape 1D long tensor (length of num_time_steps_of_predictions_matching_target_dates)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(dates_pred, torch.Tensor), "dates_pred must be a type of torch.Tensor"
    # assert isinstance(dates_target, torch.Tensor), "dates_target must be a type of torch.Tensor"

    # check which dates in predictions also exist in target dates
    matching_dates = torch.isin(dates_pred, dates_target)

    # filter the dates that also exist in target's dates
    dates_pred_filtered = dates_pred[matching_dates]

    # filter the predicted values corresponding to matching_dates
    data_pred_filtered = data_pred[:, matching_dates, :]

    # return the filtered dates and values for prediction
    return data_pred_filtered, dates_pred_filtered