# This module contains the terrestrial water storage (TWS) model

# load libraries
import torch

# terrestrial water storage
def compute_tws(swe_t: torch.Tensor, GW_t: torch.Tensor, SM_t: torch.Tensor):

    """
    This function computes terrestrial water storage at the current time step as a sum of snow water equivalent, groundwater storage and soil moisture
     at the current time steps.

    Arguments:
    swe_t: Snow water equivalent (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    GW_t: Groundwater storage (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    SM_t: Soil moisture (mm) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: tws_t containing terrestrial water storage (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(swe_t, torch.Tensor), "swe_t must be a type of torch.Tensor"
    # assert isinstance(GW_t, torch.Tensor), "GW_t must be a type of torch.Tensor"
    # assert isinstance(SM_t, torch.Tensor), "SM_t must be a type of torch.Tensor"

    # compute terrestrial water storage
    tws_t = swe_t + GW_t + SM_t

    # return terrestrial water storage
    return tws_t

# get anomalies of terrestrial water storage
def compute_tws_anomaly(tws: torch.Tensor):

    """
    This function takes the predicted (hybrid predictions) TWS (for all time steps/after the time loop) and subtracts the average value over time series for each grid. 
     After subtraction, we get anomalies which is represented by the GRACE observational TWS.

    Arguments:
    tws: The predicted (hybrid) terrestrial water storage (mm). Torch tensor of shape (batch_size, num_time_steps, 1)

    Returns: 
    tws_anomaly: The anomalies of the terrestrial water storage (mm). Torch tensor of shape (batch_size, num_time_steps, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(tws, torch.Tensor), "tws must be a type of torch.Tensor"

    # compute the average tws over time-steps for each grid in the current batch
    tws_time_mean = tws.mean(dim = (1, 2), keepdim = True) # of shape (batch_size, 1, 1)

    # subtract the mean tws from the predicted tws to get the anomalies
    tws_anomaly = tws - tws_time_mean # of shape (batch_size, num_time_steps, 1)

    # return the anomalies of terrestrial water storage
    return tws_anomaly