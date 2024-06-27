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

- `physics`: Conceptual/process models related to physical aspects.
  - `water_cycle`: Contains models for various components of the water cycle.
    - `evapotranspiration.py`, `gw_storage.py`, `runoff.py`, `snow.py`, `soil_gw_recharge.py`, `soil_moisture.py`, `tws.py`, `water_cycle_forward.py`: Models for specific water cycle processes.

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