# This module contains the snow model of the coupled water-carbon cycles

# load the libraries
import torch

# convert air temperature from Kelvin to Celsius
def convert_tair(tair_t: torch.Tensor):

    """
    This function converts air temperature's unit from Kelvin to Celsius.

    Arguments:
    tair_t: Air temperature (Kelvin) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: tair_t_celsius containing air temperature at the current time step in Celsius. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(tair_t, torch.Tensor), "tair_t must be a type of torch.Tensor"

    # convert air temperature to Celsius from Kelvin
    tair_t_celsius = tair_t - 273.15 # of shape (batch_size, 1)

    # return air temperature in Celsius
    return tair_t_celsius

# snow accumulation
def compute_snow_acc(prec_t: torch.Tensor, beta_snow: torch.Tensor, tair_t_celsius: torch.Tensor):

    """
    This function computes snow accumulation as a function of precipitation, a NN learnable parameter beta_snow and air temperature.
     Snow accumulation is modeled as precipitation if the air temperature is less than or equal to 0 celsius.

    Arguments:
    prec_t: Precipitation (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    beta_snow: A NN learnable parameter used for correction of snow. 0D torch tensor
    tair_t_celsius:  Air temperature (Celsius) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: snow_acc_t containing snow accumulations (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(prec_t, torch.Tensor), "prec_t must be a type of torch.Tensor"
    # assert isinstance(beta_snow, torch.Tensor), "beta_snow must be a type of torch.Tensor"
    # assert isinstance(tair_t_celsius, torch.Tensor), "tair_t_celsius must be a type of torch.Tensor"

    # check whether it is snowing
    # or equivalently - check whether the air temperature is less than or equal to 0
    # is_snowing will be equal to 1 if it is currently snowing, otherwise it will be 0
    is_snowing_t = torch.le(tair_t_celsius, torch.tensor(0)) * torch.tensor(1.0) # of shape (batch_size, 1)

    # compute snow accumulation
    # snow_acc will be 0, if is_snowing=0, otherwise it is precipitation multiplied by beta_snow
    snow_acc_t = (prec_t * beta_snow) * is_snowing_t # of shape (batch_size, 1)

    # return snow_acc
    return snow_acc_t

# snow melt
def compute_snow_melt(swe_t_prev: torch.Tensor, tair_t_celsius: torch.Tensor, alpha_snow_melt_t: torch.Tensor):

    """
    This function computes snow melt as a function of air temperature and a NN learned parameter alpha snow melt.
     This function indicates that if the air temperature is equal to or smaller than 0 degree celsius, then there is no snow melt.

    Arguments:
    swe_t_prev: Snow water equivalent (mm) at the previous time step t-1. Torch tensor of shape (batch_size, 1).
    tair_t_celsius: Air temperature (Celsius) at the current time step. Torch tensor of shape (batch_size, 1)
    alpha_snow_melt_t: A NN learned parameter. Torch tensor of shape (batch_size, 1)

    Returns: snow_melt_t containing the amount of snow (mm) melted at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(swe_t_prev, torch.Tensor), "swe_t_prev must be a type of torch.Tensor"
    # assert isinstance(tair_t_celsius, torch.Tensor), "tair_t_celsius must be a type of torch.Tensor"
    # assert isinstance(alpha_snow_melt_t, torch.Tensor), "alpha_snow_melt_t must be a type of torch.Tensor"

    # potential snow melt
    snow_melt_pot_t = torch.max(tair_t_celsius, torch.tensor(0)) * alpha_snow_melt_t # of shape (batch_size, 1)

    # snow melt
    snow_melt_t = torch.min(swe_t_prev, snow_melt_pot_t)

    # return the snow_melt_t parameter
    return snow_melt_t

# update snow water equivalent
def update_swe(swe_t_prev: torch.Tensor, snow_acc_t: torch.Tensor, snow_melt_t: torch.Tensor):

    """
    This function updates snow water equivalent (swe) as a function of swe at the previous time step, snow accumulation and snow melt.
     This function indicates that if the sum of swe_t_prev, snow_acc_t, and -snow_melt_t is equal to or less than 0, swe_t is 0.

    Arguments:
    swe_t_prev: Snow water equivalent (mm) at the previous time step t-1. Torch tensor of shape (batch_size, 1).
    snow_acc_t: Snow accumulation (mm) at the current time step t. Torch tensor of shape (batch_size, 1)
    snow_melt_t: Snow melt (mm) at the current time step t. Torch tensor of shape (batch_size, 1)

    Returns: Snow water equivalent (mm) at the current time step t. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(swe_t_prev, torch.Tensor), "swe_t_prev must be a type of torch.Tensor"
    # assert isinstance(snow_acc_t, torch.Tensor), "snow_acc_t must be a type of torch.Tensor"
    # assert isinstance(snow_melt_t, torch.Tensor), "snow_melt_t must be a type of torch.Tensor"

    # snow water equivalent at the current time step
    swe_t = torch.max(swe_t_prev + snow_acc_t - snow_melt_t, torch.tensor(0))

    # return the swe at the current time step
    return swe_t