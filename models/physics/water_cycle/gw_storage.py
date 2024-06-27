# This module contains the groundwater storage model.

# load libraries
import torch

# groundwater storage
def update_GW(GW_t_prev: torch.Tensor, r_gw_t: torch.Tensor, baseflow_t: torch.Tensor):

    """
    This function updates groundwater storage at the current time step by adding groundwater storage at the previous time step,
     groundwater recharge and baseflow (with a negative sign) at the current time steps.

    Arguments:
    G_t_prev: Groundwater storage (mm) at the previous time step. Torch tensor of shape (batch_size, 1)
    r_gw_t: Groundwater recharge (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    baseflow_t: Baseflow (mm) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: GW_t containing groundwater storage (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # update groundwater storage at the current time step
    GW_t = GW_t_prev + r_gw_t - baseflow_t

    # return groundwater storage at the current time step
    return GW_t