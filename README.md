# A global physically-constrained deep learning water cycle model with vegetation

This repository contains the scripts that are written for the development of the hybrid model in my PhD project.

# Folder structure of the repository:

## datasets

This folder contains the modules/scripts that are written for the dataset class that inherits Pytorch Dataset class. Specifically, the folder contains the following modules/python files:

- ZarrDataset.py: dataset class written for handling the training/testing/validation data for model training.
- helpers_loading.py: helper functions to load the data within ZarrDataset class
- helpers_preprocessing.py: helper functions to preprocess the data within ZarrDataset class

## models

This folder contains all the high level as well as lower level modules for the whole hybrid model. There are 3 subfolders:

- hybrid: collection of python modules containing high level implementation of the full hybrid model
- neural_networks: neural network part of the full hybrid model
- physics: collection of python modules containing conceptual/process model

Each subfolder contains the following modules (and/or subfolders):

- hybrid:
    - cv_helpers.py: helper functions to run k-fold cross validation
    - debug_helpers.py: helper function to debug the model (used only when there is a bug)
    - hybrid_H2O.py: the core hybrid model containing the high-level implementation using pytorch and pytorch lightning
    - hybrid_H2O_common_step.py: the common step that is used for testing/validation during training
    - hybrid_H2O_training_step.py: the training step that is used to optimise the model during training
    - hybrid_helpers.py: helper functions that are used to run the forward computation of the model
    - train_model.py: this module is used to train models using cross-validation (CV)
    - run_parallel_slurm.sh: shell script is used to allocate computational resources (GPU, memory, disk space, etc.) to jobs (tasks or processes) using slurm for 10-fold CV

- neural_networks:
    - neural_networks.py: lower-level implementation of the neural networks (used within hybrid_H2O.py)

- physics:
    - water_cycle:
        - evapotranspiration.py: evapotranspiration model
        - gw_storage.py: groundwater storage model
        - runoff.py: runoff model
        - snow.py: snow model
        - soil_gw_recharge.py: soil and groundwater recharge model
        - soil_moisture.py: soil moisture model
        - tws.py: terrestrial water storage model
        - water_cycle_forward.py: high-level forward run of full water cycle model at a single time step (used within hybrid_H2O.py)

## equifinality

This folder contains the script used to quantify the equifinality of the estimated processes after the models are trained.

- equifinality.py: module to compute the equifinality of the process estimations

# Model dependencies

We publish the following files to manage dependencies:

- requirements.txt: model dependencies for pip users
- environment.yml: model dependencies for conda users

To be able to fully reproduce and train the model the used input datasets are needed. Unfortunately we can't publish the used data, but all inputs (cited in the paper) can be downloaded from their original source.