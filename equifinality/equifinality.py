# this module contains functions to quantify equifinality of the latent predictions

# load needed libraries
import glob
import xarray as xr
from itertools import combinations
import pandas as pd

# get the full paths of a parameter across different folds
def get_file_paths(path_common: str, file_name: str):

    """
    This function returns the full file path names (recursively) that exist in the given common path.

    Arguments:
    path_common: Common path containing all the subdirectories. Replace uncommon part of the subdirectories with '**'. E.g. "a/b/**/file.txt"
    file_name: The exact name of the file whose full path name will be returned

    Returns:
    file_paths: Full paths of the files (recursive)
    """

    # get the file paths
    file_paths = glob.glob(pathname = path_common + file_name, recursive = True)

    # return the file paths as a list
    return file_paths

# get folds from the file paths
def get_fold_names(file_paths: list):

    """
    This function gets the fold names from the file paths

    Arguments:
    file_paths: A list containing the full paths of files

    Returns:
    folds: A list containing all the fold names
    """

    # make an empty list to collect folds
    folds = []


    for file_path in file_paths: # loop through each file path
        
        # split the file paths
        parts = file_path.split("/")
        
        for part in parts: # loop through each splitted part
            
            # check if the splitted part starts with the string "fold"
            if part.startswith("fold"):
                
                # if so, get the fold name as this 'part'
                fold = part
        
        # append the foldname to the list
        folds.append(fold)

    # return the final folds
    return folds

# read all files
def read_all(file_paths: list, fold_names: list, mask: xr.DataArray, testing: bool = True, mean_annual: bool = False):

    # make an empty dictionary to collect datasets
    ds_d = {}

    for idx, _ in enumerate(file_paths): # loop through each file path
        
        # read the current file (idx'th) as xarray dataset
        ds = xr.open_dataset(file_paths[idx])

        # compute mean annual
        if mean_annual and "static" not in file_paths[idx]: # only runs when mean_annual=True, and the data is not static

            # mean annual data
            ds = ds.groupby("time.year").mean(dim = "time")

        # get keep the testing grids only
        if testing:

            ds = ds.where((mask == 3).compute(), drop = True)

        # add the current dataset to the dictionary
        ds_d[fold_names[idx]] = ds

    # return the final dictionary
    return ds_d

# compute robustness between 2 datasets
def compute_robustness(ds1: xr.Dataset, ds2: xr.Dataset):

    """
    Equations are by the "Equation 5" in:

    https://www.semanticscholar.org/paper/Decomposition-of-the-mean-squared-error-and-NSE-for-Gupta-Kling/3bed3d2ae40f4b771a3b3da8498803031a49637e
    """

    # compute the mean variance between the 2 datasets
    mean_variance = (ds1.var() + ds2.var()) / 2

    ### Mean squared difference ###

    # compute the difference between the 2 datasets
    diff = ds1 - ds2

    # square the differences
    squared_diff = diff**2

    # compute the mean of the squared differences
    mean_squared_diff = squared_diff.mean()

    ###

    ### Mean difference squared ###

    # compute the difference between the means
    mean_diff = ds1.mean() - ds2.mean()

    # compute the squared difference between the means
    means_diff_squared = mean_diff**2

    ###

    ### Standard deviation ###

    # compute the standard deviations of each dataset
    std1 = ds1.std()
    std2 = ds2.std()

    # compute the difference of standard deviations
    std_diff = std1 - std2

    # compute the squared difference of standard deviations
    std_squared_diff = std_diff**2

    ###

    ### covariance ###

    # an empty dataset
    corrs = xr.Dataset()

    # loop through each dataarray bc, currently xr.corr does not support datasets
    for (_, da1), (_, da2) in zip(ds1.items(), ds2.items()): 

        # compute the correlation between 2 dataarrays
        corr = xr.corr(da1, da2)

        # add the corr value to dataset
        corrs[da1.name] = corr
    
    # compute the covariance term in the equation
    covariance = 2 * std1 * std2 * (1 - corrs)
    
    ###

    # print(f"msd: {mean_squared_diff}, total: {covariance + std_squared_diff + means_diff_squared}")

    ### normalisation ###

    # normalise the metrics
    mean_squared_diff_n = mean_squared_diff / mean_variance
    means_diff_squared_n = means_diff_squared / mean_variance
    std_squared_diff_n = std_squared_diff / mean_variance
    covariance_n = covariance / mean_variance

    ###

    # return the final (normalised) metrics
    return mean_squared_diff_n, means_diff_squared_n, std_squared_diff_n, covariance_n

# compute robustness for all the combinations of datasets in a dictionary
def compute_robustness_all(ds_dict: dict):

    # get all the combinations of the datasets in a dictionary
    ds_combinations = list(combinations(ds_dict.keys(), 2))

    # make empty lists to collect all parameters
    msd_values = []
    mds_values = []
    ssd_values = []
    covariance_values = []

    # loop through each combination
    for ds_combo in ds_combinations:

        # get the names of the current combo
        name_ds1, name_ds2 = ds_combo

        # compute the robustness for the current combo
        mean_squared_diff_n, means_diff_squared_n, std_squared_diff_n, covariance_n = compute_robustness(ds_dict[name_ds1], ds_dict[name_ds2])

        # append the current values to the corresponding lists
        msd_values.append(mean_squared_diff_n)
        mds_values.append(means_diff_squared_n)
        ssd_values.append(std_squared_diff_n)
        covariance_values.append(covariance_n)

    # return the final values
    return msd_values, mds_values, ssd_values, covariance_values

# compute the average errors
def compute_avg(values: list, name: str):

    # get the names of all parameters
    param_nms = list(values[0].keys())

    # make an empty dataframe with the shape (num_combinations, num_parameters)
    df = pd.DataFrame(columns = param_nms, index = range(len(values)))

    # loop through each parameter name
    for nm in param_nms:
        
        # an empty list to collect values separately
        l = []

        # loop through each combination mse
        for value in values:
            
            # append the current combination mse (as a numpy array) to the list
            l.append(value[nm].to_numpy())

        # overwrite the column with its corresponding values    
        df[nm] = l

    # calculate the mean error across all columns (for each parameter)
    df_mean = df.mean() # so, we get one value for each parameter

    # make a new dataframe containing the final data, with the shape (num_parameters, 2)
    df_long = pd.DataFrame({"parameter": df_mean.index, name: df_mean.values})

    # return the final dataframe
    return df_long

# full workflow to quantify equifinality for a single xarray dataset
def quantify_equifinality_full(path_common: str, file_name: str, pred_type: str, mask: xr.DataArray, testing: bool = True, mean_annual: bool = False):

    # get the full file paths
    file_paths = get_file_paths(path_common = path_common, file_name = file_name)

    # get fold names
    fold_names = get_fold_names(file_paths = file_paths)

    # read datasets and store them in a dictionary
    ds_dict = read_all(file_paths = file_paths, fold_names = fold_names, mask = mask, testing = testing, mean_annual = mean_annual)

    # compute robustness
    msd_values, mds_values, ssd_values, covariance_values = compute_robustness_all(ds_dict = ds_dict)

    # average the mse and var across combinations (of folds)
    df_msd = compute_avg(msd_values, name = "msd")
    df_mds = compute_avg(mds_values, name = "mds")
    df_ssd = compute_avg(ssd_values, name = "ssd")
    df_covariance = compute_avg(covariance_values, name = "covariance")

    # concatenate all the dataframes into a single one
    df = pd.concat([df_msd, df_mds, df_ssd, df_covariance], axis = 1)

    # the above code keeps the duplicate of the 'parameter' column, so we need to remove the duplicates
    df = df.loc[:, ~df.columns.duplicated()]

    # add a new column 'type', giving an information about the type of prediction (e.g. static, direct spatio-temporal)
    df["type"] = pred_type

    # return the final dataframe
    return df

# full workflow to quantify equifinality for multiple xarray datasets
def quantify_equifinality_full_all(path_common: str, file_names: str, pred_types: str, mask: xr.DataArray, testing: bool = True, mean_annual: bool = False):

    # an empty list to collect the dataframes
    l = []

    # loop through each file name and pred type
    for file_name, pred_type in zip(file_names, pred_types):

        # quantify the equifinality for the current dataset
        df = quantify_equifinality_full(path_common = path_common, file_name = file_name, pred_type = pred_type, mask = mask, testing = testing, mean_annual = mean_annual)

        # append the current df to the list
        l.append(df)

    # bind all dataframes (rowwise) to a single dataframe
    df_all = pd.concat(l, axis = 0, ignore_index = True)

    # return the final dataframe
    return df_all