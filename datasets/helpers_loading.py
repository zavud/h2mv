# this module contains functions to load data in

# load libraries
import xarray as xr

# load a single dataset
def load_zarr(zarr_data_path: str, data_name: str):
    
    """
    This function is a wrapper around xarray's open_zarr and is created for quickly loading datasets with more intuitive argument names (e.g. zarr_data_path).
    
    Arguments:
    zarr_data_path: the path where the zarr file is stored
    data_name: name of the dataset, that is stored in the zarr file

    Returns: a dataset
    """

    # check whether the arguments are given strings
    # assert isinstance(zarr_data_path, str), "zarr_data_path must be a string"
    # assert isinstance(data_name, str), "data_name must be a string"

    # load the data
    zarr_loaded = xr.open_zarr(store = zarr_data_path, group = data_name)
    
    # # load the data
    # if data_name != "static" and data_name != "mask": # time-series

    #     # remove any days after 2019-11-30 as the data is problematic
    #     zarr_loaded = zarr_loaded.sel(time = slice(None, "2019-11-30"))

    # return the data
    return zarr_loaded

# load multiple datasets
def load_zarr_multiple(zarr_data_path: str, data_names: list):

    """
    This is a higher level function to load multiple datasets at once using a lower level function load_zarr.

    Arguments:
    zarr_data_path: the path where the zarr file is stored
    data_name: a list of strings containing the names of the datasets to be loaded

    Returns: a list of datasets
    """

    # check whether the arguments are given the expected inputs
    # assert isinstance(zarr_data_path, str), "zarr_data_path must be a string"
    # assert isinstance(data_names, list), "data_names must be a list of the names of datasets. Use load_zarr() to load a single dataset"
    # assert all(isinstance(name, str) for name in data_names), "All values in data_names must be a string"

    # apply zarr_load to all the dataset names
    datasets_mapped = map(lambda name: load_zarr(zarr_data_path = zarr_data_path, data_name = name), data_names)

    # convert datasets_mapped into a list
    datasets_list = list(datasets_mapped)

    # return the list of datasets
    return datasets_list