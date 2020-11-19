import os, json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import shutil  
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tensorflow.keras.backend as K
import tempfile
# from IPython.display import set_matplotlib_formats
from tensorflow.python.ops import math_ops
from tensorflow.keras import layers
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude
# prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
import time
from tqdm import tqdm

from settings import *

# %matplotlib inline
assert tf.test.gpu_device_name() == '/device:GPU:0', "Not connected to GPU"

### Setup scripts

# Create folders for tracking if they are not there
# shutil.rmtree(JOB_RUNS_FOLDER)
# shutil.rmtree(TRACKER_FOLDER)
for folder in (TRACKER_FOLDER, JOB_RUNS_FOLDER):
    os.makedirs(folder, exist_ok=True)

### Functions

def job_folder(name, job):
    return os.path.join(JOB_RUNS_FOLDER, name, job)

def tracker_file(name):
    return os.path.join(TRACKER_FOLDER, name + '.json')

class JobTracker:
    
    NOT_STARTED = 'not_started'
    STARTED     = 'started'
    COMPLETED   = 'completed'
    
    def __init__(self, name):
        self.name = name
        self.location = tracker_file(name)
        self.jobs = self._create_or_read()
        
    def _create_or_read(self):
        if os.path.exists(self.location):
            print("Reading from existing location")
            with open(self.location) as f:
                return json.load(f)
        else:
            print("Creating tracker anew")
            return {}
    
    def _update_file(self):
        """ Saves current tracker to disk """
        with open(self.location, 'w') as f:
            json.dump(self.jobs, f)
        
    def create_jobs(self, job_list: list):
        """Adds an immutable list of jobs to this tracker"""
        if bool(self.jobs):
            print("Jobs have already been assigned")
        else:
            print("Creating new jobs")
            self.jobs = {job: self.NOT_STARTED for job in job_list}
            self._update_file()

            # also create one folder for each job in the JOB_RUNS_FOLDER
            # will throw an error if they already exist
            for job in job_list:
                os.makedirs(job_folder(self.name, job), exist_ok=False)
        return True
    
    def mark_as_started(self, job):
        """Marks that a specific job has started; saves to disk"""
        self.jobs[job] = self.STARTED
        self._update_file()
        return True

    def mark_as_completed(self, job):
        """Marks that a specific job has ended; saves to disk"""
        self.jobs[job] = self.COMPLETED
        self._update_file()
        return True

    def read_to_do_jobs(self) -> list:
        return [job for job, status in self.jobs.items() if status == self.NOT_STARTED]
        
    def read_unfinished_jobs(self) -> list:
        return [job for job, status in self.jobs.items() if status == self.STARTED]

    def reset_unfinished_jobs(self):
        import glob 
        """
        Does two things:
        1. For all job folders which are as STARTED, keep them but delete their contents
        2. Mark these jobs as NOT_STARTED
        """
        for job, status in self.jobs.items():
            if status == self.STARTED:
                # 1. delete contents of subfolder
                files = glob.glob(os.path.join(JOB_RUNS_FOLDER, job, "*"))
                for f in files:
                    os.remove(f)
                assert not glob.glob(os.path.join(JOB_RUNS_FOLDER, job, "*")), 'Files were not deleted upon reset'

                # 2. make job as not_started
                self.jobs[job] = self.NOT_STARTED
        
        self._update_file()
    
    def reset(self):    
                
        cont = input("This will delete all folders and models. Are you sure? [y/n]")
        if cont not in ('n', 'no', 'N', 'NO'):
            
            # Delete all folders
            for job in tqdm(self.jobs.keys()):    
                path = job_folder(self.name, job)
                if os.path.exists(path):
                    shutil.rmtree(path)

            # Delete current tracker
            os.remove(tracker_file(self.name))
            
  
    def report(self):
        total_jobs    = len(self.jobs)
        unfinished    = len(self.read_unfinished_jobs())
        not_started   = len(self.read_to_do_jobs())
        finished      = total_jobs - unfinished - not_started
        print(f"Job batch: {self.name}  -- Total jobs: {total_jobs}")
        print(f"  > Finished:    {finished} ({round(100*finished/total_jobs)} %)")
        print(f"  > Not started: {not_started} ({round(100*not_started/total_jobs)} %)")
        print(f"  > Unfinished:  {unfinished} ({round(100*unfinished/total_jobs)} %)")
  

########################################################################
##############        Data import and process   ########################
########################################################################

def read_mnist_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    return X_train, X_test, y_train, y_test

def validate_mnist_data(X, y):
    assert np.max(X.flatten()) <= 1, "Data exceeds value of 1. Preprocess again"
    assert np.min(X.flatten()) >= 0, "Data input cannot be smaller than 0. Preprocess again"
    assert np.isnan(X).sum() == 0, "Input X has NAs"
    assert np.isnan(y).sum() == 0, "Input y has NAs"
    return True

# Data process
def preprocess_mnist_data_training(X, y):
    """
    Normalize inputs and reshape into 1d vector.
    Exports normalizing parameters from training set, and will provide this to test set
    """
    max_X = np.max(X.flatten())
    min_X = np.min(X.flatten())

    new_X =  ((X-min_X)/(max_X-min_X)).reshape(X.shape[0], X.shape[1]*X.shape[2])
    new_y = y

    return new_X, new_y, max_X, min_X

def preprocess_mnist_data_test(X, y, max_X, min_X):
    """
    Normalize inputs and reshape into 1d vector
    """    
    new_X =  ((X-min_X)/(max_X-min_X)).reshape(X.shape[0], X.shape[1]*X.shape[2])
    new_y = y
    return new_X, new_y

def resize_images(X, pixel_size: int, plot=True, method='bilinear'):
    """
    Reshapes MNIST images into [pixel_size, pixel_size] images.
    This supposes we only have 1 channel
    """
    num_imgs, width, length = X.shape
    X_new = X.reshape(num_imgs, width, length, 1)
    X_new = tf.image.resize(X_new, (pixel_size, pixel_size), method=method).numpy()
    X_new = X_new.reshape(num_imgs, pixel_size, pixel_size)

    # Plot
    if plot:
        figs_to_plot = min(5, num_imgs)
        fig, ax = plt.subplots(figsize=(5,1*figs_to_plot), nrows=figs_to_plot, ncols=2)
        for i in range(figs_to_plot):
            ax[i][0].imshow(X[i,:,:])
            ax[i][1].imshow(X_new[i,:,:])
        plt.tight_layout()
        plt.show()

    return X_new

def validate_training(X, y):
    validate_mnist_data(X, y)
    return True

def validate_test(X, y):
    validate_mnist_data(X, y)
    return True

def plot_training_curves(history, metrics: list):
    n_plots = len(metrics)
    fig, ax = plt.subplots(nrows=1, ncols=n_plots, figsize=(5*n_plots,4))
    for axx, metric in zip(ax, metrics):
        axx.plot(history.history[metric], label='Train')
        axx.plot(history.history[f'val_{metric}'], label='Test')
        axx.set_title(metric)
        axx.legend()
    plt.show()

def preprocess_mnist_full(X_train_raw, X_test_raw, y_train_raw, y_test_raw, pixel_new_width):
    assert pixel_new_width in range(1,29), 'Invalid new width'

    X_train = resize_images(X_train_raw, pixel_size=pixel_new_width, plot=False)
    X_test  = resize_images(X_test_raw,  pixel_size=pixel_new_width, plot=False)

    # subsampled
    X_train, y_train, max_X, min_X = preprocess_mnist_data_training(X_train, y_train_raw)
    X_test, y_test   = preprocess_mnist_data_test(X_test, y_test_raw, max_X, min_X)

    # Data validation
    assert validate_training(X_train, y_train)
    assert validate_test(X_test, y_test)

    return X_train, X_test, y_train, y_test
    
########################################################################
##############              Training            ########################
########################################################################

class NonPos(tf.keras.constraints.Constraint):
    """Constrains the weights to be negative. Code adapted from Keras source code 
    How to call:
    model.add(layers.Dense(10, kernel_constraint=NonPos(), input_shape=(9,), activation='relu'))
    model.add(layers.Dense(10, bias_constraint=NonPos(), activation='relu'))
    """

    def __call__(self, w):
        return w * math_ops.cast(math_ops.less_equal(w, 0.), K.floatx())

class negativeHe(tf.keras.initializers.Initializer):

    def __init__(self):
#         self.seed = seed
        pass

    def __call__(self, shape, dtype=None):
        values = tf.random.uniform(shape, minval=-tf.sqrt(3.0/shape[0]), maxval=-1e-6) # seed=self.seed
        return values
    
    
def build_mnist_model(input_side_pixel: int, layer_weights: list, l2reg: float, dropout_list):

    def constrained_dense_layer(nodes, l2reg, seed=1):
            return layers.Dense(nodes, 
                            bias_constraint=NonPos(), # not allow positive bias
                            activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(l2reg),
                            bias_initializer=negativeHe() # seed
                            )
        # OBSOLETE
        # bias_initializer=tf.keras.initializers.RandomUniform( # initialize biases negatively
        #                         minval=-1.0, maxval=-0.001, seed=seed   
                            
        
    if isinstance(dropout_list, float):
        dropout_list = [dropout_list]*len(layer_weights)
    
    assert len(layer_weights) == len(dropout_list), "Layer weights and dropout list must have the same length"
    
    model = tf.keras.models.Sequential()

    # Input
    model.add(layers.Input(shape=(input_side_pixel*input_side_pixel,)))

    # Add n blocks
    for i, (n_weights, perc_dropout) in enumerate(zip(layer_weights, dropout_list)):
        model.add(constrained_dense_layer(n_weights, l2reg, seed=i))
        model.add(layers.Dropout(perc_dropout))

    # Output: keep a linear dense layer for classification
    # make sure to use from_logits=True in your loss
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'],
                )

    return model

def count_trainable_weights(model):
    return np.sum([w.numpy().size for w in model.trainable_weights])

def load_model(name, job):
    return tf.keras.models.load_model(os.path.join(job_folder(name, job), 'model'))

def flat_list_trainable_weights(model):

    weight_list = model.trainable_weights

    w_l = []
    for weight_array in weight_list:
        w_l.extend(
            list(weight_array.numpy().flatten())
        )
    
    return np.array(w_l)

def count_significative_weights(model, thresh=1e-4):

    weight_list = flat_list_trainable_weights(model)
    return len([x for x in weight_list if np.abs(x) >= thresh])

def save_baseline_model(model, X_test, y_test, job_folder_path, grid_params, training_params):
    # save model
    model.save(os.path.join(job_folder_path, 'model'))

    # save model params
    params = {'grid_params': grid_params, 'training_params': training_params}
    with open(os.path.join(job_folder_path, 'parameters.pickle'), 'wb') as f:
        pickle.dump(params, f)

    # save model performance and weights
    _, test_set_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    perf_dict = \
        {
            'test_set_accuracy':  test_set_accuracy
        }

    weights_dict = {
        'trainable_weights': float(count_trainable_weights(model)),
        'significative_weights': float(count_significative_weights(model))
    }

    print(f"  > Test set acc {round(test_set_accuracy,3)}")
    print(f"  > {weights_dict['trainable_weights']} trainable weights")
    print(f"  > {weights_dict['significative_weights']} significative weights")

    with open(os.path.join(job_folder_path, 'performance.json'), 'w') as f:
        json.dump(perf_dict, f)
    
    with open(os.path.join(job_folder_path, 'weights_info.json'), 'w') as f:
        json.dump(weights_dict, f)

    return True


def build_training_grid(**names_list_params):
    """
    Take a sequence of parameter lists and create a grid from them,
    on the form {'job_001': {param1: val1, param2: val2,....}, ...}
    """

    from itertools import product

    param_names = []
    individual_grids = []
    for param_name, values_list in names_list_params.items():
        assert isinstance(values_list, list), f"Value for key {param_name} is not a list"
        param_names.append(param_name)
        individual_grids.append(values_list)
    
    grid_list = []
    for grid in product(*individual_grids):
        grid_list.append({param_names[i]: grid[i] for i in range(len(param_names))})

    print(f"Grid has {len(grid_list)} elements")
    zfill_order = np.ceil(np.log10(len(grid_list))).astype(int)
    grid_dict = {f"job_{str(i).zfill(zfill_order)}": grid_element for i, grid_element in enumerate(grid_list)}
        
    return grid_dict

def expected_time_left(time_elapsed, current_iter, total_iters, units='mins'):
    # Find time per iteraction 
    assert current_iter > 0, 'Current iter must start from 1 and never be zero'
    avg_time = time_elapsed/current_iter

    # expected time left
    expec = avg_time * (total_iters - current_iter)

    print(f"({current_iter}/{total_iters}) Elapsed: {round(time_elapsed)},   Expected time left: {round(expec)} {units}")



if __name__ == '__main__':

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = read_mnist_data()

    grid_jobs = build_training_grid(input_side_pixel=resize_sizes, 
                                    layer_weights=base_architectures, 
                                    l2reg=reg_params,
                                    dropout_list=dropout_params)




    tracker = JobTracker(name=TRAIN_BASELINE_MODELS_JOB_ID)
    if START_FROM_SCRATCH:
        tracker.reset()
        tracker = JobTracker(name=TRAIN_BASELINE_MODELS_JOB_ID)

    # Load tracker - it will see, for this run, how many jobs have been done and so on
    # delete files and folders associated with unfinished jobs
    # then set all jobs who are incomplete to not started
    tracker.create_jobs(grid_jobs.keys())  # this will only create jobs if they haven't been defined yet
    tracker.reset_unfinished_jobs()

    # Pick subset of jobs that need to be done
    subgrid_todo_dict = {todo_job: grid_jobs[todo_job] for todo_job in tracker.read_to_do_jobs()}
    jobs_todo = len(subgrid_todo_dict)

    if jobs_todo == 0:
        print("All jobs trained")
        pass
    else:
        print(f"There are {jobs_todo} jobs left to run")
        for job_id, grid_params in tqdm(subgrid_todo_dict.items()):
            # try:
            print(f"Job: {job_id} -----------------------------------------------------------")
            tracker.mark_as_started(job_id)
            X_train, X_test, y_train, y_test = preprocess_mnist_full(X_train_raw, X_test_raw, y_train_raw, y_test_raw, 
                                                                        pixel_new_width=grid_params['input_side_pixel'])
            model = build_mnist_model(**grid_params)
            history = model.fit(X_train, y_train,
                        callbacks=[
                                    tf.keras.callbacks.EarlyStopping(
                                        monitor='val_loss', 
                                        mode='min',
                                        verbose=0, patience=20,
                                        restore_best_weights=True)],
                        **training_params)
            save_baseline_model(model, X_test, y_test, job_folder(TRAIN_BASELINE_MODELS_JOB_ID, job_id), grid_params, training_params)
            base_weights_path = os.path.join(JOB_RUNS_FOLDER, TRAIN_BASELINE_MODELS_JOB_ID, job_id, 'weights')
            model.save_weights(base_weights_path)

            # prune
            print("- Pruning...")
            for prune_thresh in pruning_grid:

                print(f"> Prune = {prune_thresh}")
                model_for_pruning = build_mnist_model(**grid_params)
                model_for_pruning.load_weights(base_weights_path).expect_partial()

                epochs_prune = pruning_training_params['epochs']
                batch_size_prune = pruning_training_params['batch_size']
                val_split_prune = pruning_training_params['validation_split']

                pruning_params = {'pruning_schedule':
                                tfmot.sparsity.keras.PolynomialDecay(
                                    initial_sparsity=0,
                                    final_sparsity=prune_thresh,
                                    begin_step=0,
                                    end_step=np.ceil(len(X_train) / batch_size_prune).astype(np.int32) * epochs_prune)
                                }
                model_for_pruning = prune_low_magnitude(model_for_pruning, **pruning_params)

                # `prune_low_magnitude` requires a recompile.
                model_for_pruning.compile(optimizer='adam',
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])

                model_for_pruning.fit(X_train, y_train,
                                    verbose=False,
                                    batch_size=batch_size_prune, epochs=epochs_prune, validation_split=val_split_prune,
                                    callbacks= [
                                        tfmot.sparsity.keras.UpdatePruningStep(),
                                        # tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
                                    ])

                # Save pruned model
                save_baseline_model(model_for_pruning, X_test, y_test, 
                                    os.path.join(job_folder(TRAIN_BASELINE_MODELS_JOB_ID, job_id), f"pruned_{prune_thresh}"), 
                                    grid_params, training_params)

            tracker.mark_as_completed(job_id)
            # except Exception as e:
            #     print(f"Could not train {grid_params}")
            #     print(e)