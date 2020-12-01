TRAIN_BASELINE_MODELS_JOB_ID = 'single_layer_jobs_20201130'
USE_GPU = True  # if True, will require there to be a local GPU
START_FROM_SCRATCH = False  # if True, will erase all jobs related to the specific run (TRAIN_BASELINE_MODELS_JOB_ID)

"""Paths """
TRACKER_FOLDER = '../simulations/trackers'  # Folders to put trackers for how many jobs were completed, and the other is to save models
JOB_RUNS_FOLDER = '../simulations/models'  # Each model should be saved to a specific folder, identified by its run identifier

"""Hyperparameter grid"""
resize_sizes = [5, 7, 10, 15]

reg_params = [0, 10 ** -3] #, 10 ** -2]
dropout_params = [0.1]


# create list of architectures based on a lot of permutations
from itertools import permutations
all_perms = []
total_len = 6
ints = list(range(2,total_len+1))
for r in ints:
    all_perms.extend([list(x) for x in permutations(ints, r=r)])

all_perms_power = []
for el in all_perms:
    all_perms_power.append([2**x for x in el])

# base_architectures = all_perms_power

base_architectures = [
    [2],
    [4],
    [6],
    [8],
    [10],
    [16],
    [20],
    [25],
    [32],
    [40],
    [50],
    [64],
    [96],
    [108],
    [128],
    [256]
]

# [
#     [2 ** 2] * 1,
#     [6] * 1,
#     [2 ** 3] * 1,
#     [2 ** 3] * 2,
#     [2 ** 3] * 3,
#     [2 ** 3] * 4,
#     [2 ** 4] * 1,
#     [2 ** 4] * 2,
#     [2 ** 4] * 3,
#     [2 ** 4] * 4,
#     [2 ** 5] * 1,
#     [2 ** 5] * 2,
#     [2 ** 5] * 3,
#     [2 ** 5] * 4,
#     [2 ** 6] * 1,
#     [2 ** 6] * 2,
#     [2 ** 6] * 3,
#     [2 ** 6] * 4,
#     [2 ** 7] * 1,
#     [2 ** 7] * 2,
#     [2 ** 7] * 3,
#     [2 ** 7] * 4,
#     [2 ** 3, 2 ** 4],
#     [2 ** 3, 2 ** 5],
#     [2 ** 3, 2 ** 6],
#     [2 ** 3, 2 ** 7],
#     [2 ** 4, 2 ** 5],
#     [2 ** 4, 2 ** 6],
#     [2 ** 4, 2 ** 7],
#     [2 ** 5, 2 ** 6],
#     [2 ** 5, 2 ** 7],
#     [2 ** 6, 2 ** 7],
#     [2 ** 7, 2 ** 7]
# ]


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
