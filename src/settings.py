#### Settings ####
TRAIN_BASELINE_MODELS_JOB_ID = 'basejobs01' # Name for batch jobs for training baseline models
PRUNE_MODELS_JOB_ID = 'pruner01'            # Name for batch jobs for pruning baseline models
TRACKER_FOLDER = 'jobs/trackers' # Folders to put trackers for how many jobs were completed, and the other is to save models
JOB_RUNS_FOLDER = 'jobs/models' # Each model should be saved to a specific folder, identified by its run identifier
START_FROM_SCRATCH = False
