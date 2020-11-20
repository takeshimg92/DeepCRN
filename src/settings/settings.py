TRAIN_BASELINE_MODELS_JOB_ID = 'include_pruning_refactor_20201119'
USE_GPU = True  # if True, will require there to be a local GPU
START_FROM_SCRATCH = False  # if True, will erase all jobs related to the specific run (TRAIN_BASELINE_MODELS_JOB_ID)

"""Paths """
TRACKER_FOLDER = '../simulations/trackers'  # Folders to put trackers for how many jobs were completed, and the other is to save models
JOB_RUNS_FOLDER = '../simulations/models'  # Each model should be saved to a specific folder, identified by its run identifier

"""Hyperparameter grid"""
resize_sizes = [5, 7, 10, 15]
base_architectures = [
    [2 ** 2] * 1,
    [6] * 1,
    [2 ** 3] * 1,
    [2 ** 3] * 2,
    [2 ** 3] * 3,
    [2 ** 3] * 4,
    [2 ** 4] * 1,
    [2 ** 4] * 2,
    [2 ** 4] * 3,
    [2 ** 4] * 4,
    [2 ** 5] * 1,
    [2 ** 5] * 2,
    [2 ** 5] * 3,
    [2 ** 5] * 4,
    [2 ** 6] * 1,
    [2 ** 6] * 2,
    [2 ** 6] * 3,
    [2 ** 6] * 4,
    [2 ** 7] * 1,
    [2 ** 7] * 2,
    [2 ** 7] * 3,
    [2 ** 7] * 4
    [2 ** 3, 2 ** 4],
    [2 ** 3, 2 ** 5],
    [2 ** 3, 2 ** 6],
    [2 ** 3, 2 ** 7],
    [2 ** 4, 2 ** 5],
    [2 ** 4, 2 ** 6],
    [2 ** 4, 2 ** 7],
    [2 ** 5, 2 ** 6],
    [2 ** 5, 2 ** 7],
    [2 ** 6, 2 ** 7],
    [2 ** 7, 2 ** 7]
]
reg_params = [0, 10 ** -3, 10 ** -2]
dropout_params = [0.1]

training_params = \
    {'verbose': False,
     'epochs': 40,
     'validation_split': 0.1,
     'batch_size': 128
     }

pruning_training_params = \
    {'verbose': False,
     'epochs': 6,
     'validation_split': 0.1,
     'batch_size': 100
     }

pruning_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
