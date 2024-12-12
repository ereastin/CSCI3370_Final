## [CSCI3370] Final Project Fall 2024
### "Re-evaluating Central U.S. Precipitation via Multiscale Interactions in an Inception-based U-Net Model"

#### Training & testing models:

`python train.py <model_name> <run_type> <n_epochs>`

##### Options for `<model_name> (str)`:
1. "inc": Inception-ResNet-v4 U-Net model (ours)
2. "aunet": Attention U-Net (Zhang et al.)
3. "unet": U-Net (Zhang et al.)
4. "cnn": CNN (Zhang et al.)

##### Options for `<run_type> (str)`:
1. "train", trains the specificed model for specified number of epochs
2. "test", tests the specified model and plots figures

##### Options for `<n_epochs> (int)`:
1. 0 - when paired with "train" this runs a hyperparameter grid search
2. 1 - when paired with "train" this will time a single epoch 
3. Any integer. Specifying "test" ignores this value

#### Preprocessing:

`python preprocess.py <year> <month>`

This is parallelized via slurm.

