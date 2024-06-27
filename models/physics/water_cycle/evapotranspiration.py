# This module contains the snow model of the coupled water-carbon cycles

# load libraries
import torch

# Rainfall
def compute_rainfall(prec_t: torch.Tensor, tair_t_celsius: torch.Tensor):

    """
    This function computes rainfall as a function of precipitation and air temperature. Rainfall is simply the precipitation, if the air temperature is greater than 0. 
     Otherwise, it is equal to 0.

    Arguments:
    prec_t: Precipitation (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    tair_t_celsius: Air temperature (Celsius) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: rainfall_t containing the amount of rainfall (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(prec_t, torch.Tensor), "prec_t must be a type of torch.Tensor"
    # assert isinstance(tair_t_celsius, torch.Tensor), "tair_t must be a type of torch.Tensor"

    # find whether the weather condition for the rain is met (air temp > 0 Celsius)
    # days where air temp is greater than 0 will have a value of 1, otherwise 0
    is_raining = torch.greater(tair_t_celsius, torch.tensor(0)) * torch.tensor(1.0) # of shape (batch_size, 1)

    # compute rainfall - which is simply precipitation if is_raining=1, otherwise 0
    rainfall_t = prec_t * is_raining # of shape (batch_size, 1)

    # return the computed rainfall
    return rainfall_t

# interception evaporation
def compute_Ei(rainfall_t: torch.Tensor, fAPAR_nn_t: torch.Tensor, rn_t_mm: torch.Tensor, alpha_Ei_t: torch.Tensor):

    """
    This function computes interception evaporation as a function of rainfall, fAPAR and a NN learned parameter alpha_Ei.

    Arguments:
    rainfall_t: The amount of rainfall (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    fAPAR_nn_t: The fraction of photosynthetically active radiation (400-700 nm) absorbed by green vegetation (predicted) at the current time step. Torch tensor of shape (batch_size, 1)
    rn_t_mm: Net radiation (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    alpha_Ei_t: A NN learned parameter used for modelling interception evaporation. Torch tensor of shape (batch_size, 1)

    Returns: Ei_t containing interception evaporation (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(rainfall_t, torch.Tensor), "rainfall_t must be a type of torch.Tensor"
    # assert isinstance(fAPAR_nn_t, torch.Tensor), "fAPAR_nn_t must be a type of torch.Tensor"
    # assert isinstance(alpha_Ei_t, torch.Tensor), "alpha_Ei_t must be a type of torch.Tensor"
    # assert isinstance(rn_t_mm, torch.Tensor), "rn_t_mm must be a type of torch.Tensor"

    # compute ~maximum water holding capacity (max_whc_t) of plants
    max_whc_t = fAPAR_nn_t * alpha_Ei_t # of shape (batch_size, 1)

    # compute potential interception evaporation (Ei) as a minimum of rainfall and max_whc of plants
    Ei_pot_t = torch.minimum(rainfall_t, max_whc_t) # of shape (batch_size, 1)

    # compute interception evaporation (Ei) as a minimum of potential Ei and the available energy
    Ei_t = torch.minimum(Ei_pot_t, rn_t_mm)

    # return interception evaporation
    return Ei_t

# convert rn from Watts per square meter per day to mm per day
def convert_rn_to_mm(rn_t: torch.Tensor):

    """
    This function converts Net radiation's unit from Watts per square meter per day to mm per day.

    Please see: https://www.researchgate.net/post/How_to_convert_30minute_evapotranspiration_in_watts_to_millimeters

    Arguments:
    rn_t: Net radiation (Watts / m^2) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: rn_t_mm containing the Net radiation (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(rn_t, torch.Tensor), "rn_t must be a type of torch.Tensor"

    # convert net radiation from (Watts / m^2) to (MJ / m^2)
    rn_t_MJ = rn_t * 0.0864

    # convert net radiation's unit (rn_t) from (MJ / m^2) to mm (depth).
    rn_t_mm = rn_t_MJ / 2.45 # of shape (batch_size, 1)

    # return net radiation with a unit of mm
    return rn_t_mm

# convert negative values of rn_t_mm to 0
def make_zero_if_negative(rn_t_mm: torch.Tensor):

    """
    This function converts negative values of net radiation (mm) to 0 for physical computation (not used as an input).

    Arguments:
    rn_t_mm: Net radiation (mm) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: rn_t_mm_converted containing the Net radiation (mm) negative values converted to 0s at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(rn_t_mm, torch.Tensor), "rn_t must be a type of torch.Tensor"

    # convert the values of rn_t_mm to 0 if they are negative
    rn_t_mm_converted = torch.maximum(rn_t_mm, torch.tensor(0.0))

    # return the converted tensor
    return rn_t_mm_converted

# update net radiation
def update_Rn(rn_t_mm: torch.Tensor, water_flux_t: torch.Tensor):

    """
    This function subtracts interception evaporation (Ei) or soil evaporation (Es) from net radiation (Rn).

    rn_t_mm: Net radiation (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    water_flux_t: Interception evaporation or soil evaporation (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns:
    rn_t_mm: Updated net radiation (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(rn_t_mm, torch.Tensor), "rn_t_mm must be a type of torch.Tensor"
    # assert isinstance(water_flux_t, torch.Tensor), "water_flux_t must be a type of torch.Tensor"

    # update net radiation by subtracting Ei from it
    rn_t_mm = rn_t_mm - water_flux_t

    # return the updated net radiation
    return rn_t_mm


# compute potential evapotranspiration
def compute_ET_pot(rn_t_mm: torch.Tensor, SM_t: torch.Tensor):

    """
    This function computes potential evapotranspiration as a function of net radiation and soil moisture.

    Arguments:
    rn_t_mm: Net radiation (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    SM_t: Soil moisture (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    
    Returns: ET_pot_t containing potential evapotranspiration (mm/day). Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(rn_t_mm, torch.Tensor), "rn_t_mm must be a type of torch.Tensor"
    # assert isinstance(SM_t, torch.Tensor), "SM_t must be a type of torch.Tensor"

    # compute potential evapotranspiration as a minimum of net radiation (mm) and soil moisture
    ET_pot_t = torch.minimum(rn_t_mm, SM_t)

    # return the potential evapotranspiration
    return ET_pot_t

# soil evaporation
def compute_Es(rn_t_mm: torch.Tensor, fAPAR_nn_t: torch.Tensor, SM_t: torch.Tensor, alpha_Es_t: torch.Tensor, beta_Es: torch.Tensor):

    """
    This function computes soil evaporation (Es) as a function of net radiation, fAPAR, soil moisture, NN learned parameters 
     alpha_Es (varied in time and space) and beta_Es (constant).

    Arguments:
    rn_t_mm: Net radiation (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    fAPAR_nn_t: The fraction of photosynthetically active radiation (400-700 nm) absorbed by green vegetation (predicted) at the current time step. Torch tensor of shape (batch_size, 1)
    SM_t: Soil moisture (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    alpha_Es_t: A NN learned parameter used to compute Es (varies both in time and space). Torch tensor of shape (batch_size, 1)
    beta_Es: A NN learned parameter used to compute Es (globally constand). Torch tensor of shape 0D (scalar)

    Returns: Es_t containing soil evaporation (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(rn_t_mm, torch.Tensor), "rn_t_mm must be a type of torch.Tensor"
    # assert isinstance(fAPAR_nn_t, torch.Tensor), "fAPAR_nn_t must be a type of torch.Tensor"
    # assert isinstance(SM_t, torch.Tensor), "SM_t must be a type of torch.Tensor"
    # assert isinstance(alpha_Es_t, torch.Tensor), "alpha_Es_t must be a type of torch.Tensor"
    # assert isinstance(beta_Es, torch.Tensor), "beta_Es must be a type of torch.Tensor"

    # compute/model soil coverage as approx. being (1 - vegetation) in the grid
    soil_coverage = 1 - fAPAR_nn_t # of shape (batch_size, 1)

    # compute Es using current net radiation (rn)
    Es_rn_t = rn_t_mm * soil_coverage * alpha_Es_t # of shape (batch_size, 1)

    # compute Es using current soil moisture (SM)
    Es_SM_t = SM_t * beta_Es # of shape (batch_size, 1)

    # compute the final Es as a minimum of Es_rn and Es_SM
    Es_t = torch.minimum(Es_rn_t, Es_SM_t) # of shape (batch_size, 1)

    # return the final soil evaporation
    return Es_t

# soil evaporation 2 (proposed by Markus on 29-11-22: testing phase for now)
def compute_Es2(rn_t_mm: torch.Tensor, fAPAR_nn_t: torch.Tensor, SM_t: torch.Tensor, alpha_Es_t: torch.Tensor, alpha_Es_supply: torch.Tensor):

    """
    This function computes soil evaporation (Es) as a function of net radiation, fAPAR, soil moisture, NN learned parameters 
     alpha_Es (varied in time and space) and alpha_Es_supply.

    Arguments:
    rn_t_mm: Net radiation (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    fAPAR_nn_t: The fraction of photosynthetically active radiation (400-700 nm) absorbed by green vegetation (predicted) at the current time step. Torch tensor of shape (batch_size, 1)
    SM_t: Soil moisture (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    alpha_Es_t: A NN learned parameter used to compute Es (varies both in time and space). Torch tensor of shape (batch_size, 1)
    alpha_Es_supply: A NN learned parameter used to compute Es (static). Torch tensor of shape (batch_size, 1)

    Returns: Es_t containing soil evaporation (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(rn_t_mm, torch.Tensor), "rn_t_mm must be a type of torch.Tensor"
    # assert isinstance(fAPAR_nn_t, torch.Tensor), "fAPAR_nn_t must be a type of torch.Tensor"
    # assert isinstance(SM_t, torch.Tensor), "SM_t must be a type of torch.Tensor"
    # assert isinstance(alpha_Es_t, torch.Tensor), "alpha_Es_t must be a type of torch.Tensor"
    # assert isinstance(alpha_Es_supply, torch.Tensor), "alpha_Es_supply must be a type of torch.Tensor"

    # compute/model soil coverage as approx. being (1 - vegetation) in the grid
    soil_coverage = 1 - fAPAR_nn_t # of shape (batch_size, 1)

    # compute Es_demand
    Es_demand_t = rn_t_mm * soil_coverage * alpha_Es_t # of shape (batch_size, 1)

    # compute Es_supply
    Es_supply_t = SM_t * alpha_Es_supply # of shape (batch_size, 1)

    # compute the final Es as a minimum of Es_demand and Es_supply
    Es_t = torch.minimum(Es_demand_t, Es_supply_t) # of shape (batch_size, 1)

    # return the final soil evaporation
    return Es_t

# soil evaporation 3 (new implementation)
def compute_Es3(fAPAR_nn_t: torch.Tensor, ET_pot_t: torch.tensor, alpha_Es_t: torch.tensor):

    """
    This function computes soil evaporation as a function of fapar, potential ET and two NN learned parameters.

    Arguments:
    fAPAR_nn_t: The fraction of photosynthetically active radiation (400-700 nm) absorbed by green vegetation (predicted) at the current time step. Torch tensor of shape (batch_size, 1)
    ET_pot_t:  Potential evapotranspiration (mm/day). Torch tensor of shape (batch_size, 1)
    alpha_Es_t: NN learned parameter varied both in space and time. Torch tensor of shape (batch_size, 1)
    
    Returns: Es_t containing soil evaporation. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(fAPAR_nn_t, torch.Tensor), "fAPAR_nn_t must be a type of torch.Tensor"
    # assert isinstance(ET_pot_t, torch.Tensor), "ET_pot_t must be a type of torch.Tensor"
    # assert isinstance(alpha_Es_t, torch.Tensor), "alpha_Es_t must be a type of torch.Tensor"

    # compute/model soil coverage as approx. being (1 - vegetation) in the grid
    soil_coverage_t = 1 - fAPAR_nn_t # of shape (batch_size, 1)

    # compute potential soil evaporation
    Es_pot_t = soil_coverage_t * ET_pot_t

    # compute final soil evaporation using NN learned parameters
    Es_t = Es_pot_t * alpha_Es_t

    # return the final soil evaporation
    return Es_t


# transpiration demand
def compute_T_demand(rn_t_mm: torch.Tensor, fAPAR_nn_t: torch.Tensor, alpha_T_demand_t: torch.Tensor):

    """
    This function computes Transpiration demand as a function of net radiation (mm), fAPAR and NN learned parameter alpha_T_demand.

    Arguments:
    rn_t_mm: Net radiation (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    fAPAR_nn_t: The fraction of photosynthetically active radiation (400-700 nm) absorbed by green vegetation (predicted) at the current time step. Torch tensor of shape (batch_size, 1)
    alpha_T_demand_t: A NN learned parameter used to model Transpiration demand. Torch tensor of shape (batch_size, 1)

    Returns: T_demand_t (mm) containing Transpiration demand at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(rn_t_mm, torch.Tensor), "rn_t_mm must be a type of torch.Tensor"
    # assert isinstance(fAPAR_nn_t, torch.Tensor), "fAPAR_nn_t must be a type of torch.Tensor"
    # assert isinstance(alpha_T_demand_t, torch.Tensor), "alpha_T_demand_t must be a type of torch.Tensor"

    # compute transpiration demand
    T_demand_t = rn_t_mm * fAPAR_nn_t * alpha_T_demand_t # of shape (batch_size, 1)

    # return transpiration demand
    return T_demand_t


# transpiration supply
def compute_T_supply(SM_t: torch.Tensor, alpha_T_supply_t: torch.Tensor):

    """
    This function computes Transpiration supply as a function of Soil moisture and a NN learned parameter alpha_T_supply.

    Arguments:
    SM_t: Soil moisture (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    alpha_T_supply_t: A NN learned parameter used to compute transpiration supply. Torch tensor of shape (batch_size, 1)

    Returns: T_supply_t containing transpiration supply (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(SM_t, torch.Tensor), "SM_t must be a type of torch.Tensor"
    # assert isinstance(alpha_T_supply_t, torch.Tensor), "alpha_T_supply_t must be a type of torch.Tensor"

    # compute transpiration supply
    T_supply_t = SM_t * alpha_T_supply_t # of shape (batch_size, 1)

    # return transpiration supplye
    return T_supply_t

# transpiration
def compute_T(T_demand_t: torch.Tensor, T_supply_t: torch.Tensor):

    """
    This function computes transpiration at the current time step as a minimum between transpiration demand and supply.

    Arguments:
    T_demand_t: Transpiration demand (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    T_supply_t: Transpiration supply (mm) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: T_t containing transpiration (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(T_demand_t, torch.Tensor), "T_demand_t must be a type of torch.Tensor"
    # assert isinstance(T_supply_t, torch.Tensor), "T_supply_t must be a type of torch.Tensor"

    # model transpiration
    T_t = torch.minimum(T_demand_t, T_supply_t) # of shape (batch_size, 1)

    # return transpiration
    return T_t

# transpiration 2
def compute_T2(fAPAR_nn_t: torch.Tensor, ET_pot_t: torch.Tensor, alpha_T_t: torch.Tensor):

    """
    This function computes transpiration as a function of fapar, potential ET and two NN learned parameters.

    Arguments:
    fAPAR_nn_t: The fraction of photosynthetically active radiation (400-700 nm) absorbed by green vegetation (predicted) at the current time step. Torch tensor of shape (batch_size, 1)
    ET_pot_t:  Potential evapotranspiration (mm/day). Torch tensor of shape (batch_size, 1)
    alpha_T_t: NN learned parameter varied both in space and time. Torch tensor of shape (batch_size, 1)
    
    Returns: Es_t containing soil evaporation. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(fAPAR_nn_t, torch.Tensor), "fAPAR_nn_t must be a type of torch.Tensor"
    # assert isinstance(ET_pot_t, torch.Tensor), "ET_pot_t must be a type of torch.Tensor"
    # assert isinstance(alpha_T_t, torch.Tensor), "alpha_T_t must be a type of torch.Tensor"

    # compute/model plant coverage as approx. being vegetation/fapar in the grid
    plant_coverage_t = fAPAR_nn_t # of shape (batch_size, 1)

    # compute potential soil evaporation
    T_pot_t = plant_coverage_t * ET_pot_t

    # compute final soil evaporation using NN learned parameters
    T_t = T_pot_t * alpha_T_t

    # return the final soil evaporation
    return T_t

# evapotranspiration
def compute_ET(Ei_t: torch.Tensor, Es_t: torch.Tensor, T_t: torch.Tensor):

    """
    This function computes evapotranspiration as a sum of interception evaporation, soil evaporation and transpiration at the
     current time step.

    Arguments:
    Ei_t: Interception evaporation (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    Es_t: Soil evaporation (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    T_t: Transpiration (mm) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: ET_t containing evapotranspiration (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    """
    # check whether the types of the given arguments are correct
    # assert isinstance(Ei_t, torch.Tensor), "Ei_t must be a type of torch.Tensor"
    # assert isinstance(Es_t, torch.Tensor), "Es_t must be a type of torch.Tensor"
    # assert isinstance(T_t, torch.Tensor), "T_t must be a type of torch.Tensor"

    # compute evapotranspiration
    ET_t = Ei_t + Es_t + T_t # of shape (batch_size, 1)

    # return evapotranspiration
    return ET_t

# evaporative fraction
def compute_evaporative_frac(ET_t: torch.Tensor, rn_t_mm: torch.Tensor):

    """
    This function computes evaporative fraction at the current time step. Note that, if the net radiation is 0, the function is undefined. In this case, evaporative fraction is set to 0.

    Arguments:
    ET_t: Evapotranspiration (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    rn_t_mm: Net radiation (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: EF_t containing evaporative fraction at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(ET_t, torch.Tensor), "ET_t must be a type of torch.Tensor"
    # assert isinstance(rn_t_mm, torch.Tensor), "rn_t_mm must be a type of torch.Tensor"

    # compute evaporative fraction
    EF_t = ET_t / rn_t_mm # of shape (batch_size, 1)

    # set EF_t to 0 in case net radiation is 0 (bc, in pytorch division by zero equals to Inf)
    EF_t[rn_t_mm == torch.tensor(0.0)] = torch.tensor(0.0) # of shape (batch_size, 1)

    # return evapotranspiration
    return EF_t