# This module contains functions to be used for preprocessing of the data, mostly within ZarrDataset class

# load libraries
import numpy as np
import xarray as xr

# get_non_zero_indices
def get_non_zero_indices(mask: xr.core.dataset.Dataset):

    """
    This function is just a wrapper around the numpy's argwhere.
    It is created here to give it a more informative/intuitive name and for tidy coding within ZarrDataset class.

    Arguments:
    mask: xarray dataset containing the values of 0's for the masked pixels/grids.

    Returns: a 2D array containing the indices of the coordinates that are not masked (or equal to 0) of shape (num_non_zero_indices, 2)
    """

    # check whether the mask has the right data format
    # assert isinstance(mask, xr.core.dataset.Dataset), "mask must be xarray Dataset (xarray.core.dataset.Dataset)"

    # find indices of the coordinates that are not masked (or are non-zero)
    non_zero_indices = np.argwhere(mask.data.values)

    # return the indices of the coords that are not masked (or non-zero)
    return non_zero_indices

# get_forcing
def get_forcing(forcing_list_xrdatasets: list, index_lat: np.int64, index_lon: np.int64):

    """
    This function takes a list of xarray Datasets and indices of latitude and longitude and
     gives a time series data of the selected grid/pixel.

    Arguments:
    forcing_list_xrdatasets: a list of (forcing) xarray Datasets
    index_lat: the selected index of latitude
    index_lon: the selected index of longitude

    Returns: a numpy array containing the time series data for the selected grid of shape (num_time_steps, num_forcing_variables)
    """

    # check if the given arguments have expected types
    # assert isinstance(forcing_list_xrdatasets, list), "forcing_list_xrdatasets must be a list of xarray Datasets"
    # assert all(isinstance(dataset, xr.core.dataset.Dataset) for dataset in forcing_list_xrdatasets), \
    #     "all datasets in forcing_list_xrdatasets must be xr.core.dataset.Dataset"
    # assert isinstance(index_lat, np.int64), "index_lat must be numpy.int64"
    # assert isinstance(index_lon, np.int64), "index_lat must be numpy.int64"

    # get time-series of forcing variables for the selected grid and store it in a list
    grid_time_series_l = [dataset.isel(lat = index_lat, lon = index_lon).data.values.astype("float32") for dataset in forcing_list_xrdatasets]

    # stack the time series of forcing variables as a 2D numpy array
    grid_time_series_2Darray = np.stack(grid_time_series_l, axis = -1) # of shape (num_time_steps, num_forcing_variables)

    # return the final data
    return grid_time_series_2Darray

# get_constraints
def get_constraint(constraint: xr.core.dataset.Dataset, index_lat: np.int64, index_lon: np.int64):

    """
    This function takes a constraint as an xarray dataset, indices of the selected latitutde and longitute and returns the corresponding
     time series data for the selected grid.

    Arguments:
    constraint: a constraint as an xarray dataset
    index_lat: the selected index of latitude
    index_lon: the selected index of longitude
    """

    # check if the given arguments have expected types
    # assert isinstance(constraint, xr.core.dataset.Dataset), "constraint must be xr.core.dataset.Dataset"
    # assert isinstance(index_lat, np.int64), "index_lat must be numpy.int64"
    # assert isinstance(index_lon, np.int64), "index_lat must be numpy.int64"

    # get the constraint time-series values
    constraint_values = constraint.isel(lat = index_lat, lon = index_lon).data.values.astype("float32")

    # get the dates on which the constraint has values
    # .astype("datetime64[M]") - converts from yyyy/mm/dd to yyyy/mm
    # .astype(np.int64) - gives a unique integer to each year-month combination
    # .astype(np.int64) - is done because pytoch tensors does not support datetime64 format
    # dates are needed during the model cost function computation
    constraint_dates = constraint.isel(lat = index_lat, lon = index_lon).time.values.astype("datetime64[M]").astype(np.int64)

    # return constraint values and dates as a tuple
    return constraint_values, constraint_dates

# get_static
def get_static(static: xr.core.dataset.Dataset, index_lat: np.int64, index_lon: np.int64):

    """
    This function takes the static data as an xarray Dataset, indices of latitude and longitude, and returns the static values for the selected grid.

    Arguments:
    static: static data as an xarray dataset
    index_lat: the selected index of latitude
    index_lon: the selected index of longitude

    Returns: a numpy array containing the values of static variables shape of (num_static_variables,)
    """

    # check if the given arguments have expected types
    # assert isinstance(static, xr.core.dataset.Dataset), "static must be xr.core.dataset.Dataset"
    # assert isinstance(index_lat, np.int64), "index_lat must be numpy.int64"
    # assert isinstance(index_lon, np.int64), "index_lon must be numpy.int64"

    # get the values of static data for the selected grid
    static_values = static.isel(lat = index_lat, lon = index_lon).data.values.astype("float32") # shape of (num_static_variables,)

    # return the static values
    return static_values

# combine_features
def combine_features(forcing: np.ndarray, static: np.ndarray):

    """
    This function takes forcing and static data of a single grid as numpy arrays and combines them. Broadcasting is applied to 
     static data to make the dimensions compatible with forcing data. For details, see the code below.

    Arguments:
    forcing: numpy array of forcing data for a single grid of shape (num_time_steps, num_forcing_variables)
    static: numpy array of static data for a single grid of shape (num_static_variables,)

    Returns: a combined features as a numpy array of shape (num_time_steps, num_forcing_variables + num_static_variables)
    """

    # check whether the gives inputs are the expected type
    # assert isinstance(forcing, np.ndarray), "forcing must be np.ndarray"
    # assert isinstance(static, np.ndarray), "static must be np.ndarray"

    # broadcast the static variables so that its dimensions match with forcing
    static_repeated = np.broadcast_to(static, shape = (forcing.shape[0], static.shape[0])) # of shape (num_time_steps, num_static_variables)

    # combine the repeated static data and forcing data
    features_combined = np.concatenate((forcing, static_repeated), axis = -1) # of shape (num_time_steps, num_forcing_variables + num_static_variables)

    # return the combined data
    return features_combined

# get the exact coordinates of the selected grid
def get_coords(dataset: xr.core.dataset.Dataset, index_lat: np.int64, index_lon: np.int64):

    """
    This function takes the indices of the latitude and longitude as inputs and returns the exact corresponding latitudes and longitudes.

    Arguments:
    dataset: Any xarray dataset containing the input data
    index_lat: the selected index of latitude
    index_lon: the selected index of longitude

    Returns: lat, lon containing the exact latitude and longitude of the selected grid. Both are numpy arrays (0D)
    """

    # check if the given arguments have expected types
    # assert isinstance(dataset, xr.core.dataset.Dataset), "dataset must be xr.core.dataset.Dataset"
    # assert isinstance(index_lat, np.int64), "index_lat must be numpy.int64"
    # assert isinstance(index_lon, np.int64), "index_lon must be numpy.int64"

    # get the latitude
    lat = dataset.isel(lat = index_lat, lon = index_lon).coords["lat"].values

    # get the longitude
    lon = dataset.isel(lat = index_lat, lon = index_lon).coords["lon"].values

    # return the exact latitude and longitude
    return lat, lon

# quick & dirty split into train/val/test sets
def random_split(coord_indices: np.ndarray):

    """
    This function is a quick data split into train/val/test sets. It takes as an argument the coordinate indices of the global data.

    Arguments:
    coord_indices: Coordinate indices of the global data. Numpy array of (num_examples, num_coordinates)

    Returns: training, validation, testing sets. All of are numpy arrays of shape (7692, 2), (1649, 2), (1649, 2)
    """

    # set the seed so that training, validation and testing sets will always be the same
    np.random.seed(0)

    # get the training indices as 70% of the whole input data (70% = 7692)
    indices_training = np.random.choice(coord_indices.shape[0], size = 7692, replace = False)

    # store training data with the randomly selected grids
    training = coord_indices[indices_training]

    # remove the training data from the full data
    coord_indices = np.delete(coord_indices, indices_training, axis = 0)

    # get the validation indices as 15% of full data or half of the rest of the data (15% of full data = half of the rest = 1649)
    indices_validation = np.random.choice(coord_indices.shape[0], size = 1649, replace = False)

    # store validation data
    validation = coord_indices[indices_validation]

    # remove validation data/store testing data
    testing = np.delete(coord_indices, indices_validation, axis = 0)

    # return all the three data sets
    return training, validation, testing

# Spatial blocking
def blocks_like(xr_array, num_sets, block_size):
    """
    Creates randomly blocked xr.DataArray with given block_size from 1-``num_stripes``. 
    
    (NOTE: This function was provided by Basil Kraft)
    """

    nlat = len(xr_array.lat)
    nlon = len(xr_array.lon)

    nlat_blocks = np.ceil(nlat / block_size).astype(int)
    nlon_blocks = np.ceil(nlon / block_size).astype(int)

    # Do lon increasing block.
    a = np.arange(nlon_blocks)
    a = np.tile(a, (nlat_blocks, 1))

    # Do lat increasing block.
    b = (np.arange(0, nlat_blocks) * nlon_blocks).reshape(-1, 1)
    b = b.repeat(nlon_blocks, axis=1)

    # Combine lona & lat, repeat to create blocks.
    b = (a + b).repeat(block_size, axis=0).repeat(block_size, axis=1)

    # Increment blocks to cv sets.
    b_unique = np.unique(b)
    blocks = np.zeros_like(b)
    np.random.shuffle(b_unique)
    sets = np.array_split(b_unique, num_sets)

    for i, set in enumerate(sets):
        blocks[np.isin(b, set)] = i + 1

    # Make xr.Dataset.
    m = xr.Dataset({'data': xr.DataArray(blocks, coords=[
                   xr_array.lat, xr_array.lon], dims=['lat', 'lon'])})

    m = m.where(xr_array, 0)

    return m

# get folds 
def get_folds(i, num_folds=10):

    """
    This function gets the folds where approx. 80% of grids go to training, 10% go to validation and 10% go to testing sets
    
    (NOTE: This function was provided by Basil Kraft)
    """

    validation = {i}
    test = {(i + 1) % num_folds}
    training = set(np.arange(num_folds) % num_folds) - validation - test

    training = np.asarray(list(training)) + 1
    validation = np.asarray(list(validation)) + 1
    test = np.asarray(list(test)) + 1

    return training, validation, test

# get folds for k-fold cross validation
def get_folds_k(k: int, num_folds: int, get_testing: bool):

    """
    This function gets folds for validation, test and training sets when k-fold cross validation technique is used.

    Arguments:
    k: k'th cross-validation fold

    Returns:
    A tuple of fold_training, fold_validation, fold_testing containing corresponding training, validation and testing folds
    """

    # make an array containing all the folds (fold numbers)
    folds = np.arange(start = 1, stop = num_folds + 1) # start - inclusive, stop - exclusive (that's why add 1 to num_folds)

    if get_testing:

        # choose the 1st fold as testing set
        fold_testing = np.asarray(folds[0])

        # the rest is used for training (which is then split into training/validation)
        folds = folds[folds != fold_testing] # remove testing fold

    # get the k'th validation fold
    fold_validation = np.asarray(folds[k])

    # remove the validation fold from the training fold
    fold_training = folds[folds != fold_validation]

    # return folds for training and testing
    if get_testing:

        return fold_training, fold_validation, fold_testing
    
    else:

        return fold_training, fold_validation

# extract a set of grids for testing
def get_test_set(mask_xr_array):

    """
    This function extracts about 16.6% of the original global data as a testing set to later test equifinality among the trained multiple models.

    Arguments:
    mask_xr_array: A mask of the global dataset (xr.DataArray)

    Returns:
    (testing_ds, remaining_ds): A tuple containing testing and remaining (training+validation) sets
    """

    # get the blocked mask as xr.Dataset
    mask_blocked = blocks_like(mask_xr_array, num_sets = 13, block_size = 5) # num_sets = 6 will give 7 folds, because 0 means "invalid"

    # get the folds for training, validation and testing sets
    # here k does not matter, as the testing set is always the same
    # num_folds = 6 is used because in this case, testing set will be approximately ~10% of the original data
    fold_training, fold_validation, fold_testing = get_folds_k(k = 1, num_folds = 13, get_testing = True)

    # get the testing set as a dataset
    testing_ds = mask_blocked.isin(fold_testing)

    # get the remaining (training + validation) set as dataset
    remaining_ds = mask_blocked.isin(np.concatenate([fold_training, fold_validation.reshape(1)]))

    # return the testing and remaining sets
    return testing_ds, remaining_ds

# random split using spatial blocking
def random_split_blocking(mask_xr_array):

    """
    This function splits the data into train/val/test (80%/10%/10%) sets using the spatial blocking technique in order to reduce the problem of spatial
     autocorrelation.
    
    Arguments:
    mask_xr_array: A mask of the global dataset (xr.DataArray)

    Returns: 
    A tuple of coords_training, coords_validation, coords_testing containing the corresponding coordinates
    """

    # set a random seed, so that training/validation/testing sets are always the same
    np.random.seed(1)

    # get the blocked mask as xr.Dataset
    mask_blocked = blocks_like(mask_xr_array, num_sets = 10, block_size = 5)

    # get the folds for training, validation and testing sets
    fold_training, fold_validation, fold_testing = get_folds(0, num_folds = 10)

    # get the masks for training, validation and testing sets as xr.Dataset
    mask_training = mask_blocked.isin(fold_training)
    mask_validation = mask_blocked.isin(fold_validation)
    mask_testing = mask_blocked.isin(fold_testing)

    # get the coordinates for training, validation and testing sets as xr.Dataset
    coords_training = get_non_zero_indices(mask_training)
    coords_validation = get_non_zero_indices(mask_validation)
    coords_testing = get_non_zero_indices(mask_testing)

    return coords_training, coords_validation, coords_testing

# random split using spatial blocking
def random_split_blocking_k(mask_xr_array, num_folds: int, k: int, return_back: str):

    """
    This function splits the data into train/val/test (80%/10%/10%) sets using the spatial blocking technique in order to reduce the problem of spatial
     autocorrelation. It should only be used when k-fold cross validation technique is used
    
    Arguments:
    mask_xr_array: A mask of the global dataset (xr.DataArray)
    k: k'th cross-validation fold

    Returns: 
    A tuple of coords_training, coords_validation, coords_testing containing the corresponding coordinates
    """

    # set a random seed, so that training/validation/testing sets are always the same
    np.random.seed(1)

    # get a separate testing set
    mask_testing, mask_remaining = get_test_set(mask_xr_array = mask_xr_array)

    # get the blocked mask as xr.Dataset
    mask_blocked = blocks_like(mask_remaining.data, num_sets = num_folds, block_size = 5) # num_sets = 10 will give 11 folds, because 0 means "invalid"

    # get the folds for training, validation and testing sets
    fold_training, fold_validation = get_folds_k(k = k, num_folds = num_folds, get_testing = False)

    # get the masks for training, validation and testing sets as xr.Dataset
    mask_training = mask_blocked.isin(fold_training)
    mask_validation = mask_blocked.isin(fold_validation)

    # return the coordinates
    if return_back == "masks":

        return mask_training.data, mask_validation.data, mask_testing.data # return the masks as DataArrays
    
    elif return_back == "coords":

        # get the coordinates for training, validation and testing sets as xr.Dataset
        coords_training = get_non_zero_indices(mask_training)
        coords_validation = get_non_zero_indices(mask_validation)
        coords_testing = get_non_zero_indices(mask_testing)

        return coords_training, coords_validation, coords_testing