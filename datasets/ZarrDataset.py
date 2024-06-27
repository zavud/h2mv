# this module contains the ZarrDataset class

# load libraries
from torch.utils.data import Dataset

# load the custom modules
import helpers_loading as hl # helper functions to load the data
import helpers_preprocessing as hp # helper functions to preprocess the data

class ZarrDataset(Dataset): # the custom dataset must inherit from torch.utils.data.Dataset

    def __init__(self, zarr_data_path: str, data_split: str, k: int):

        super().__init__() # initialize the parent class Dataset

        # store the names of the forcing data in a list
        forcing_names = ["rn", "prec", "tair"]

        # store the names of the constraints in a list
        self.constraints_names = ["tws", "et", "q", "swe", "fapar"]

        # load the forcing data in
        self.forcing_xr = hl.load_zarr_multiple(zarr_data_path, forcing_names)

        # load the constraints in
        self.constraints_xr = hl.load_zarr_multiple(zarr_data_path, self.constraints_names)
        # self.constraints_static_xr = hl.load_zarr_multiple(zarr_data_path, self.constraints_names_static)

        # load the static data in
        self.static_xr = hl.load_zarr(zarr_data_path, "static")

        # load the mask in
        self.mask = hl.load_zarr(zarr_data_path, "mask")

        # get the indices of the grids that are not masked (or are non-zero)
        self.coord_indices = hp.get_non_zero_indices(self.mask)

        if data_split != "global": # use all the grids for global inference

            # split the input data into train/val/test sets using spatial blocking
            training, validation, testing = hp.random_split_blocking_k(mask_xr_array = self.mask.data, num_folds = 10, k = k, return_back = "coords")

            if data_split == "training":

                self.coord_indices = training
        
            elif data_split == "validation":

                self.coord_indices = validation
        
            else: # testing

                self.coord_indices = testing

    def __len__(self):

        # find the number of grids (or examples) in the dataset
        num_grids = self.coord_indices.shape[0]

        # return the number of grids in the dataset
        return num_grids

    def __getitem__(self, index):

        # get the index'th latitude and longitude
        index_lat, index_lon = self.coord_indices[index]

        # get the exact coordinates of the selected grid
        lat, lon = hp.get_coords(dataset = self.mask, index_lat = index_lat, index_lon = index_lon) # of shape (64,) -> only when returned by __getitem__()

        # get the time series of forcing data for the selected (or index'th) grid
        forcing = hp.get_forcing(self.forcing_xr, index_lat, index_lon) # of shape (num_time_steps, num_forcing_variables)

        # get the static input variables
        static = hp.get_static(self.static_xr, index_lat, index_lon) # of shape (num_static_variables,)

        # get the time series of constraints data for the selected (or index'th) grid as a dictionary of tuples (e.g. (constraint, dates))
        # each constraint and its corresponding date has a shape of (num_time_steps_of_the_constraint,)
        constraints_values_dates = {self.constraints_names[idx]: hp.get_constraint(constraint, index_lat, index_lon) for idx, constraint in enumerate(self.constraints_xr)}

        # return the final results
        return forcing, static, constraints_values_dates, (lat, lon)