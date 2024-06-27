## This module contains functions needed to implement training using cross validation

# import the needed libraries
import hybrid_H2O as hybrid_model
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import dask

# training loop
def training_loop(dir_trained_models: str, dir_name_new_model: str, zarr_data_path: str, k: int, train_data_loader = None, val_data_loader = None, checkpoints_resume = None):

    """
    This function starts training the model untill convergence using the k'th cross validation fold.
    
    """

    # the following is needed to enable multiple number of workers to preprocess the data
    dask.config.set(scheduler='synchronous')

    # define logger of the metrics
    csv_logger = CSVLogger(
        save_dir = dir_trained_models,
        name = dir_name_new_model
        )
    
    # instantiate the hybrid water cycle model
    model = hybrid_model.HybridH2O(zarr_data_path = zarr_data_path, device = "cuda", k = k)

    # define the checkpoint callback for the model
    checkpoint_callback = ModelCheckpoint(dirpath = dir_trained_models + dir_name_new_model + "/best_model", save_top_k=1, monitor="loss_sum_validation")

    # define the early stopping callback
    early_stopping_callback = EarlyStopping('loss_sum_validation', patience = 30)

    # define the trainer
    trainer = pl.Trainer(max_epochs = 5000, 
                   accelerator="gpu", 
                   devices=1, 
                   detect_anomaly = False,
                   log_every_n_steps=1, 
                   logger=[csv_logger],
                   callbacks=[checkpoint_callback, early_stopping_callback],
                   gradient_clip_val=0.5, 
                   gradient_clip_algorithm = "value")
    
    # start the training
    if checkpoints_resume: # if checkpoints are given, then resume from it

        if train_data_loader and val_data_loader: # if the train_data_loader & val_data_loader are both None ...

            trainer.fit(model = model, train_dataloaders = train_data_loader, val_dataloaders = val_data_loader, ckpt_path = checkpoints_resume) # ... then run the model with the default training and validation data loaders defined within the model

        else: # otherwise ...

            trainer.fit(model = model, ckpt_path = checkpoints_resume) # ... run it using the given data-loaders

    else:

        if train_data_loader and val_data_loader: # if the train_data_loader & val_data_loader are both None ...

            trainer.fit(model = model, train_dataloaders = train_data_loader, val_dataloaders = val_data_loader) # ... then run the model with the default training and validation data loaders defined within the model

        else: # otherwise ...

            trainer.fit(model = model) # ... run it using the given data-loaders