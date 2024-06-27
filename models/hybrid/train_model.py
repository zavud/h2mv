import cv_helpers as cv_h
import argparse

# Create an argument parser
parser = argparse.ArgumentParser()
parser.add_argument('k', type=int, help='Parameter value')

# Parse the command-line arguments
args = parser.parse_args()

# Access the parameter value
k = args.k

# store the model directory
dir_trained_models = "..." # path where the trained models will be saved to
dir_name_new_model = "10_cv"

# path to the zarr file containing inputs and outputs
zarr_data_path = "..."

cv_h.training_loop(dir_trained_models=dir_trained_models, dir_name_new_model=dir_name_new_model + "_fold" + str(k+1), zarr_data_path=zarr_data_path, k = k)