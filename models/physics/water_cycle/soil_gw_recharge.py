# This module contains the model of soil recharge and groundwater recharge fractions

# load libraries
import torch

# water input
def compute_water_input(rainfall_t: torch.Tensor, snow_melt_t: torch.Tensor, Ei_t: torch.Tensor):

    """
    This function computes water input as a sum of rainfall, snow melt and interception evaporation (with a negative sign).

    Arguments:
    rainfall_t: Rainfall (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    snow_melt_t: Amount of snow melted (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    Ei_t: Interception evaporation (mm) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: water_input containing water input (mm). Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(rainfall_t, torch.Tensor), "rainfall_t must be a type of torch.Tensor"
    # assert isinstance(snow_melt_t, torch.Tensor), "snow_melt_t must be a type of torch.Tensor"
    # assert isinstance(Ei_t, torch.Tensor), "Ei_t must be a type of torch.Tensor"

    # computr water input
    water_input_t = rainfall_t + snow_melt_t - Ei_t # of shape (batch_size, 1)

    # return water input
    return water_input_t

# soil recharge fraction
def compute_r_soil_fraction(SM_t: torch.Tensor, SM_max_nn: torch.Tensor, alpha_r_soil: torch.Tensor):

    """
    This function computes soil recharge fraction soil moisture at the previous time step, a NN learned parameters maximum soil water capacity and alpha_r_soil.

    SM_t: Soil moisture (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    SM_max_nn: A NN learned parameter maximum soil water capacity (mm) varied only spatially (fixed in time). Torch tensor of shape (batch_size, 1)
    alpha_r_soil: A NN learned parameter varied only spatially (fixed in time). Torch tensor of shape (batch_size, 1)

    Returns: r_soil_t_fraction containing soil recharge fraction at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(SM_t, torch.Tensor), "SM_t must be a type of torch.Tensor"
    # assert isinstance(SM_max_nn, torch.Tensor), "SM_max_nn must be a type of torch.Tensor"
    # assert isinstance(alpha_r_soil, torch.Tensor), "alpha_r_soil must be a type of torch.Tensor"

    # small epsilon value to prevent nominator from becoming 0
    epsilon = torch.tensor(1e-8) # this is necessary otherwise derivative of the following equation can become undefined (nan) if, e.g. alpha_r_soil=1/2

    # compute soil recharge fraction
    r_soil_t_fraction = 1 - ((torch.maximum(SM_t, epsilon) / SM_max_nn)**alpha_r_soil)

    # return soil recharge fraction at the current time step
    return r_soil_t_fraction

# soil recharge fraction 2
def compute_r_soil_fraction2(SM_t: torch.Tensor, SM_max_nn: torch.Tensor, water_input_t: torch.Tensor, alpha_r_soil_t: torch.Tensor):

    """
    This function computes soil recharge fraction as a function of soil moisture at the previous time step,
      a NN learned parameters maximum soil water capacity, water input and alpha_r_soil.

    SM_t: Soil moisture (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    SM_max_nn: A NN learned parameter maximum soil water capacity (mm) varied only spatially (fixed in time). Torch tensor of shape (batch_size, 1)
    water_input_t: Water input (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    alpha_r_soil_t: A NN learned parameter. Torch tensor of shape (batch_size, 1)

    Returns: r_soil_t_fraction containing soil recharge fraction at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(SM_t, torch.Tensor), "SM_t must be a type of torch.Tensor"
    # assert isinstance(SM_max_nn, torch.Tensor), "SM_max_nn must be a type of torch.Tensor"
    # assert isinstance(water_input_t, torch.Tensor), "water_input_t must be a type of torch.Tensor"
    # assert isinstance(alpha_r_soil_t, torch.Tensor), "alpha_r_soil_t must be a type of torch.Tensor"

    # small epsilon value to prevent nominator from becoming 0
    epsilon = torch.tensor(1e-8) # this is necessary otherwise derivative of the following equation can become undefined (nan) if, e.g. alpha_r_soil_t=1/2

    # compute current soil moisture deficit
    SM_deficit_t = SM_max_nn - SM_t

    # compute the ratio of SM deficit to water input
    ratio_deficit2win_t = SM_deficit_t / torch.maximum(water_input_t, epsilon)

    # compute potential soil recharge fraction
    r_soil_fraction_pot_t = torch.minimum(torch.tensor(1.0), ratio_deficit2win_t)

    # compute soil recharge fraction
    r_soil_fraction_t = r_soil_fraction_pot_t * alpha_r_soil_t

    # return soil recharge fraction at the current time step
    return r_soil_fraction_t

# soil recharge
def compute_r_soil(water_input_t: torch.Tensor, SM_t: torch.Tensor, SM_max_nn: torch.Tensor, r_soil_t_fraction: torch.Tensor):

    """
    This function computes soil recharge as a function of water input at the current time step, soil moisture at the current time step, 
     a NN learned parameters maximum soil water capacity and alpha_r_soil.

    Arguments:
    water_input_t: Water input (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    SM_t: Soil moisture (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    SM_max_nn: A NN learned parameter maximum soil water capacity (mm) varied only spatially (fixed in time). Torch tensor of shape (batch_size, 1)
    r_soil_t_fraction: Soil recharge fraction (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: A tuple of the following: 
    r_soil_t: containing soil recharge (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    r_soil_remaining_water_t: containing the remaining water (mm/day) in case r_soil_candidate_t is bigger than sm_deficit_t.
     Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(water_input_t, torch.Tensor), "water_input_t must be a type of torch.Tensor"
    # assert isinstance(SM_t, torch.Tensor), "SM_t must be a type of torch.Tensor"
    # assert isinstance(SM_max_nn, torch.Tensor), "SM_max_nn must be a type of torch.Tensor"
    # assert isinstance(r_soil_t_fraction, torch.Tensor), "r_soil_t_fraction must be a type of torch.Tensor"

    # compute the soil moisture deficit
    sm_deficit_t = SM_max_nn - SM_t

    # compute soil recharge (flux) candidate
    r_soil_candidate_t = r_soil_t_fraction * water_input_t

    # soil recharge cannot be greater than the current soil mositure deficit
    r_soil_t = torch.minimum(r_soil_candidate_t, sm_deficit_t)

    # in case sm_deficit_t is smaller than r_soil_candidate_t, there is remaining water
    #  that needs to be partitioned into groundwater recharge and surface runoff
    r_soil_remaining_water_t = r_soil_candidate_t - r_soil_t

    # return soil recharge
    return r_soil_t, r_soil_remaining_water_t

# soil recharge 2
def compute_r_soil2(water_input_t: torch.Tensor, r_soil_t_fraction: torch.Tensor):

    """
    This function computes soil recharge as a function of water input at the current time step and soil recharge fraction

    Arguments:
    water_input_t: Water input (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    r_soil_t_fraction: Soil recharge fraction (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns:
    r_soil_t: containing soil recharge (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(water_input_t, torch.Tensor), "water_input_t must be a type of torch.Tensor"
    # assert isinstance(r_soil_t_fraction, torch.Tensor), "r_soil_t_fraction must be a type of torch.Tensor"

    # compute soil recharge (flux)
    r_soil_t = r_soil_t_fraction * water_input_t

    # return soil recharge
    return r_soil_t

# groundwater recharge
def compute_r_gw(water_input_t: torch.Tensor, r_soil_remaining_water_t: torch.Tensor, SM_oveflow_t: torch.Tensor, r_soil_t_fraction: torch.Tensor, alpha_r_gw_t: torch.Tensor):

    """
    This function computes ground water recharge as a function of water input, soil recharge and a NN learned parameter alpha_r_gw_t.

    Arguments:
    water_input_t: Water input (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    r_soil_remaining_water_t: The remaining water (mm/day) that cannot enter to the soil in case r_soil_candidate_t is bigger than sm_deficit_t.
     Torch tensor of shape (batch_size, 1)
    SM_oveflow_t: Overflow of water from soil at the current time step. Torch tensor of shape (batch_size, 1)
    r_soil_t_fraction: Soil recharge fraction at the current time step. Torch tensor of shape (batch_size, 1)
    alpha_r_gw_t: NN learned parameter used to model groundwater recharge. It is between 0 and 1. Torch tensor of shape (batch_size, 1)

    Returns: r_gw_t containing ground water recharge (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(water_input_t, torch.Tensor), "water_input_t must be a type of torch.Tensor"
    # assert isinstance(r_soil_remaining_water_t, torch.Tensor), "r_soil_remaining_water_t must be a type of torch.Tensor"
    # assert isinstance(SM_oveflow_t, torch.Tensor), "SM_oveflow_t must be a type of torch.Tensor"
    # assert isinstance(r_soil_t_fraction, torch.Tensor), "r_soil_t_fraction must be a type of torch.Tensor"
    # assert isinstance(alpha_r_gw_t, torch.Tensor), "alpha_r_gw_t must be a type of torch.Tensor"

    # compute groundwater recharge fraction
    r_gw_t_fraction = (1 - r_soil_t_fraction) * alpha_r_gw_t

    # compute groundwater recharge
    r_gw_t = r_gw_t_fraction * (water_input_t + r_soil_remaining_water_t + SM_oveflow_t)

    # return groundwater recharge
    return r_gw_t

# compute fraction of groundwater recharge
def compute_r_gw_frac(r_soil_t_fraction: torch.Tensor, alpha_r_gw_t: torch.Tensor):

    """
    This function computes the fraction of water input that will go to groundwater to recharge it.

    Arguments:
    r_soil_t_fraction: Soil recharge fraction at the current time step. Torch tensor of shape (batch_size, 1)
    alpha_r_gw_t: NN learned parameter used to model groundwater recharge. It is between 0 and 1. Torch tensor of shape (batch_size, 1)

    Returns:
    r_gw_t_fraction_t: Fraction of groundwater recharge (-/day) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # compute groundwater recharge fraction
    r_gw_t_fraction_t = (1 - r_soil_t_fraction) * alpha_r_gw_t

    # return groundwater recharge fraction
    return r_gw_t_fraction_t

# groundwater recharge 2
def compute_r_gw2(water_input_t: torch.Tensor, r_gw_t_fraction_t: torch.Tensor):

    """
    This function computes ground water recharge as a function of water input, soil recharge fraction and a NN learned parameter alpha_r_gw_t.

    Arguments:
    water_input_t: Water input (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    r_gw_t_fraction_t: Fraction of groundwater recharge (-/day) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: r_gw_t containing ground water recharge (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(water_input_t, torch.Tensor), "water_input_t must be a type of torch.Tensor"
    # assert isinstance(r_gw_t_fraction_t, torch.Tensor), "r_gw_t_fraction_t must be a type of torch.Tensor"

    # compute groundwater recharge
    r_gw_t = r_gw_t_fraction_t * water_input_t

    # return groundwater recharge
    return r_gw_t