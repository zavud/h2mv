# Global Physically-Constrained Deep Learning Water Cycle Model with Vegetation

Welcome to the GitHub repository for our hybrid hydrological model with vegetation (H2MV). H2MV represents a hybrid approach, combining deep learning with physical constraints to simulate the water cycle and its interaction with vegetation.

## Repository Structure

This repository is organized into several key folders, each containing specific types of scripts and modules necessary for the model's development and operation.

### `datasets`

This directory hosts scripts and modules for managing datasets, specifically designed to work with the PyTorch Dataset class. It includes:

- `ZarrDataset.py`: Manages training, testing, and validation datasets for model training.
- `helpers_loading.py`: Provides helper functions for loading data into the ZarrDataset class.
- `helpers_preprocessing.py`: Contains helper functions for preprocessing data within the ZarrDataset class.

### `models`

Contains all modules related to the model's architecture, divided into three subdirectories:

- `hybrid`: Includes Python modules for the high-level implementation of the hybrid model.
  - `cv_helpers.py`: Functions for running k-fold cross-validation.
  - `debug_helpers.py`: Debugging functions (used for bug fixes).
  - `hybrid_H2O.py`: Core hybrid model implementation using PyTorch and PyTorch Lightning.
  - `hybrid_H2O_common_step.py`: Common steps for testing/validation during training.
  - `hybrid_H2O_training_step.py`: Training steps for model optimization.
  - `hybrid_helpers.py`: Functions for forward computation of the model.
  - `train_model.py`: Module for model training using cross-validation.
  - `run_parallel_slurm.sh`: Shell script for allocating computational resources via Slurm for 10-fold cross-validation.

- `neural_networks`: Lower-level neural network implementations.
  - `neural_networks.py`: Implementation details of neural networks within the hybrid model.

- `physics`: This section delves into the physical processes and conceptual models that underpin the hybrid model's understanding of the water cycle and its interactions. Each module within this directory is dedicated to a specific aspect of the water cycle.
  - `evapotranspiration.py`: Simulates the process of evapotranspiration, integrating both the evaporation from land, interception evaporation and the transpiration from plants.
  - `gw_storage.py`: Models the groundwater storage dynamics.
  - `runoff.py`: Captures the runoff processes, including both surface runoff and baseflow.
  - `snow.py`: Represents the snowpack dynamics, including accumulation, melting.
  - `soil_gw_recharge.py`: Simulates the recharge of soil and groundwater
  - `soil_moisture.py`: Models soil moisture dynamics, critical for understanding plant-water interactions, evapotranspiration, and soil water storage.
  - `tws.py`: Stands for Terrestrial Water Storage, encompassing three components of water storage on land, including snow, soil moisture, and groundwater.
  - `water_cycle_forward.py`: Provides a high-level forward run of the complete water cycle model at a single timestep, integrating the various components and processes modeled in the other modules.

### `equifinality`

- `equifinality.py`: Script for quantifying the equifinality of estimated processes post-training.

## Model Dependencies

To ensure full reproducibility and ease of setup, we provide two files for managing dependencies:

- `requirements.txt`: For pip users.
- `environment.yml`: For Conda users.

**Note:** While the input datasets used for training the model are not included in this repository, all referenced datasets in our paper can be obtained from their original sources.

## Getting Started

To get started with this model, please ensure you have the necessary dependencies installed by following the instructions in either `requirements.txt` or `environment.yml`. Due to the complexity and computational requirements of the model, access to appropriate hardware and computational resources is recommended.

## Contributing and Support

We warmly welcome contributions, feedback, and questions! If you encounter any issues, have suggestions for improvements, or want to contribute to the project, please feel free to open an issue in the issues section of this repository. Your input is valuable in making this model more robust and useful for the community.