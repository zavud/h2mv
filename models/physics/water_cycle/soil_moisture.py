# This module contains the model of soil moisture

# load libraries
import torch

# update soil moisture
def update_SM(SM_current_or_prev: torch.Tensor, r_soil_t: torch.Tensor, T_t: torch.Tensor, Es_t: torch.Tensor):

    """
    This function updates soil moisture at the current time step by adding soil moisture at the previous or current time step, soil recharge,
     transpiration and soil evaporation (with a negative sign) at the current time step.

    Arguments:
    SM_current_or_prev: Soil moisture (mm) at the previous or current time step. Torch tensor of shape (batch_size, 1)
    r_soil_t: Soil recharge (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    T_t: Transpiration (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    Es_t: Soil evaporation (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: SM_t containing soil moisture (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(SM_current_or_prev, torch.Tensor), "SM_current_or_prev must be a type of torch.Tensor"
    # assert isinstance(r_soil_t, torch.Tensor), "r_soil_t must be a type of torch.Tensor"
    # assert isinstance(T_t, torch.Tensor), "T_t must be a type of torch.Tensor"
    # assert isinstance(Es_t, torch.Tensor), "Es_t must be a type of torch.Tensor"

    # update soil moisture at the current time step
    SM_t = SM_current_or_prev + r_soil_t - T_t - Es_t

    # return soil moisture at the current time step
    return SM_t

# compute soil moisture overflow
def compute_SM_overflow(SM_t: torch.Tensor, SM_max_nn: torch.Tensor):

    """
    This function computes the overflow of water from soil. This ensures that current soil moisture cannot exceed the maximum water holding capacity of soil.

    Arguments:
    SM_t: Soil moisture (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    SM_max_nn: A NN predicted parameter maximum soil water capacity (mm) varied only spatially (fixed in time). Torch tensor of shape (batch_size, 1)

    Returns: SM_oveflow_t containing overflow of water from soil at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(SM_t, torch.Tensor), "SM_t must be a type of torch.Tensor"
    # assert isinstance(SM_max_nn, torch.Tensor), "SM_max_nn must be a type of torch.Tensor"

    # compute overflow
    # note that, if SM_max_nn is bigger than SM_t SM_oveflow_t is 0
    SM_oveflow_t = torch.maximum(SM_t - SM_max_nn, torch.tensor(0.0))

    # return the soil moisture overflow
    return SM_oveflow_t

# remove overflow from soil moisture
def remove_SM_overflow(SM_t: torch.Tensor, SM_oveflow_t: torch.Tensor):

    """
    This function removes the overflow of water from soil moisture.

    Arguments:
    SM_t: Soil moisture (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    SM_oveflow_t: Overflow of water from soil at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: SM_t_overflow_removed containing soil moisture from which overflow of water was removed.
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(SM_t, torch.Tensor), "SM_t must be a type of torch.Tensor"
    # assert isinstance(SM_oveflow_t, torch.Tensor), "SM_oveflow_t must be a type of torch.Tensor"

    # remove overflow from the soil moisture
    SM_t_overflow_removed = SM_t - SM_oveflow_t

    # return the new soil moisture
    return SM_t_overflow_removed

# relative soil moisture
def compute_relative_SM(SM_t: torch.Tensor, SM_max_nn: torch.Tensor):

    """
    This function computes relative soil moisture (or, soil moisture fraction) at the current time step.

    Arguments:
    SM_t: Soil moisture (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    SM_max_nn: A NN predicted parameter maximum soil water capacity (mm) varied only spatially (fixed in time). Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(SM_t, torch.Tensor), "SM_t must be a type of torch.Tensor"
    # assert isinstance(SM_max_nn, torch.Tensor), "SM_max_nn must be a type of torch.Tensor"

    # compute relative soil moisture
    rel_SM_t = SM_t / SM_max_nn

    # return the relatove soil moisture content
    return rel_SM_t