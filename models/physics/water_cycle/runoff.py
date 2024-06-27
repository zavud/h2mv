# This module contains the runoff model.

# load libraries
import torch

# surface runoff
def compute_runoff_surface(water_input_t: torch.Tensor, r_soil_remaining_water_t: torch.Tensor, SM_oveflow_t: torch.Tensor, r_soil_t_fraction: torch.Tensor, alpha_r_gw_t: torch.Tensor):

    """
    This function computes surface runoff as a function of water input, soil recharge and a NN learned parameter alpha_r_gw_t.

    Arguments:
    water_input_t: Water input (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    r_soil_remaining_water_t: The remaining water (mm/day) that cannot enter to the soil in case r_soil_candidate_t is bigger than sm_deficit_t.
     Torch tensor of shape (batch_size, 1)
    SM_oveflow_t: Overflow of water from soil at the current time step. Torch tensor of shape (batch_size, 1)
    r_soil_t_fraction: Soil recharge fraction at the current time step. Torch tensor of shape (batch_size, 1)
    alpha_r_gw_t: NN learned parameter used to model groundwater recharge. It is between 0 and 1. Torch tensor of shape (batch_size, 1)

    Returns: runoff_surface_t containing surface runoff (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(water_input_t, torch.Tensor), "water_input_t must be a type of torch.Tensor"
    # assert isinstance(r_soil_remaining_water_t, torch.Tensor), "r_soil_remaining_water_t must be a type of torch.Tensor"
    # assert isinstance(SM_oveflow_t, torch.Tensor), "SM_oveflow_t must be a type of torch.Tensor"
    # assert isinstance(r_soil_t_fraction, torch.Tensor), "r_soil_t_fraction must be a type of torch.Tensor"
    # assert isinstance(alpha_r_gw_t, torch.Tensor), "alpha_r_gw_t must be a type of torch.Tensor"

    # compute surface runoff (fraction)
    runoff_surface_fraction_t = (1 - r_soil_t_fraction) * (1 - alpha_r_gw_t)

    # compute surface runoff
    runoff_surface_t = runoff_surface_fraction_t * (water_input_t + r_soil_remaining_water_t + SM_oveflow_t)

    # return surface runoff
    return runoff_surface_t

# surface runoff fraction
def runoff_surface_frac(r_soil_t_fraction: torch.Tensor, alpha_r_gw_t: torch.Tensor):

    """
    This function computes surface runoff fraction.
    
    Arguments:
    r_soil_t_fraction: Soil recharge fraction at the current time step. Torch tensor of shape (batch_size, 1)
    alpha_r_gw_t: NN learned parameter used to model groundwater recharge. It is between 0 and 1. Torch tensor of shape (batch_size, 1)

    Returns:
    runoff_surface_fraction_t: Surface runoff fraction (-/day) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # compute surface runoff (fraction)
    runoff_surface_fraction_t = (1 - r_soil_t_fraction) * (1 - alpha_r_gw_t)

    # return the surface runoff fraction
    return runoff_surface_fraction_t


# surface runoff
def compute_runoff_surface2(water_input_t: torch.Tensor, runoff_surface_fraction_t: torch.Tensor):

    """
    This function computes surface runoff at the current time step.

    Arguments:
    water_input_t: Water input (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    runoff_surface_fraction_t: Surface runoff fraction (-/day) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: runoff_surface_t containing surface runoff (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(water_input_t, torch.Tensor), "water_input_t must be a type of torch.Tensor"
    # assert isinstance(runoff_surface_fraction_t, torch.Tensor), "runoff_surface_fraction_t must be a type of torch.Tensor"

    # compute surface runoff
    runoff_surface_t = runoff_surface_fraction_t * water_input_t

    # return surface runoff
    return runoff_surface_t

# baseflow
def compute_baseflow(GW_t_prev: torch.Tensor, beta_baseflow: torch.Tensor):

    """
    This function computes baseflow as a product between groundwater storage at the previous time step and a NN learned scaler beta_baseflow.

    Arguments:
    GW_t_prev: Groundwater storage (mm) at the previous time step. Torch tensor of shape (batch_size, 1)
    beta_baseflow: A NN learned static parameter (fixed in time and but varied in space). Torch tensor of shape (batch_size, 1)

    Returns: baseflow_t containing baseflow (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    """
    # check whether the types of the given arguments are correct
    # assert isinstance(GW_t_prev, torch.Tensor), "GW_t_prev must be a type of torch.Tensor"
    # assert isinstance(beta_baseflow, torch.Tensor), "beta_baseflow must be a type of torch.Tensor"

    # compute baseflow
    baseflow_t = GW_t_prev * beta_baseflow

    # return baseflow
    return baseflow_t

# total runoff
def compute_runoff_total(runoff_surface_t: torch.Tensor, baseflow_t: torch.Tensor):

    """
    This function computes the total runoff as a sum of surface runoff and baseflow at the current time step.

    Arguments:
    runoff_surface_t: Surface runoff (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    baseflow_t: Baseflow (mm) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: runoff_total_t containing total runoff (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(runoff_surface_t, torch.Tensor), "runoff_surface_t must be a type of torch.Tensor"
    # assert isinstance(baseflow_t, torch.Tensor), "baseflow_t must be a type of torch.Tensor"

    # compute total runoff
    runoff_total_t = runoff_surface_t + baseflow_t

    # return total runoff
    return runoff_total_t