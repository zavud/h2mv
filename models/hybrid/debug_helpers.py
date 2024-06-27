# this module contains helper functions that are helpful for debugging the hybrid model

# load libraries
import torch

def retain_grads_all(dict_tensors_list: dict):

    """
    This function retains the gradients of all the intermediate tensors. 

    Arguments:
    dict_tensors_list: A dictionary containing lists of tensors at each time step. Each list is a modelled parameter

    Returns: The function does not return anything and uses an in-place operation.
    """

    for data_list in dict_tensors_list.values(): # loop through each list in the dictionary

        for time_step in data_list: # loop through each time step

            # retain the gradients of all time steps
            time_step.retain_grad()


def clip_physics_gradients(dict_list_tensor: dict, clip_value: torch.Tensor):

    """
    This function clips the gradients of physical layers.

    Arguments:
    dict_tensors_list: A dictionary containing lists of tensors at each time step. Each list is a modelled parameter
    clip_value: A maximum value allowed for the gradients of physical layers. (e.g. -clip_value, clip_value)
    """

    for data_list in dict_list_tensor.values(): # loop through each list in the dictionary

        for time_step in data_list: # loop through each time step

            torch.nn.utils.clip_grad_value_(time_step, clip_value)