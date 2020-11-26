import os
import tensorflow as tf
from tqdm import tqdm

from settings.settings import *
from job_scheduler.jobtracker import JobTracker, job_folder
from pipeline.mnist_pipeline import read_mnist_data, preprocess_mnist_full
from training.model_utils import build_training_grid, build_mnist_model, save_baseline_model, base_weights_path, prune_model


if __name__ == '__main__':

    if USE_GPU:
        assert tf.test.gpu_device_name() == '/device:GPU:0', "Not connected to GPU"

    for folder in (TRACKER_FOLDER, JOB_RUNS_FOLDER):
        os.makedirs(folder, exist_ok=True)

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = read_mnist_data()

    grid_jobs = build_training_grid(input_side_pixel=resize_sizes,
                                    layer_weights=base_architectures,
                                    l2reg=reg_params,
                                    dropout_list=dropout_params)

    tracker = JobTracker(name=TRAIN_BASELINE_MODELS_JOB_ID)
    if START_FROM_SCRATCH:
        tracker.reset()
        tracker = JobTracker(name=TRAIN_BASELINE_MODELS_JOB_ID)

    tracker.create_jobs(grid_jobs.keys())  # this will only create jobs if they haven't been defined yet
    tracker.reset_unfinished_jobs()
    tracker.report()

    # Pick subset of jobs that need to be done
    subgrid_todo_dict = {todo_job: grid_jobs[todo_job] for todo_job in tracker.read_to_do_jobs()}; jobs_todo = len(subgrid_todo_dict)


    def train_worker(data, params):
        grid_params, train_params = break_params(params)
        X_train, X_test, y_train, y_test = break_data(data)
        X_train, X_test, y_train, y_test = preprocess_mnist_full(X_train, X_test, y_train, y_test,
                                                                 pixel_new_width=grid_params['input_side_pixel'])
        


    if jobs_todo == 0:
        print("All jobs trained")
        pass
    else:
        print(f"There are {jobs_todo} jobs left to run")
        for job_id, grid_params in tqdm(subgrid_todo_dict.items()):

            """
            Train baseline model
            """
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
            save_baseline_model(model, X_test, y_test,
                                job_folder(TRAIN_BASELINE_MODELS_JOB_ID, job_id),
                                grid_params,
                                training_params)

            model.save_weights(base_weights_path(job_id, JOB_RUNS_FOLDER, TRAIN_BASELINE_MODELS_JOB_ID))

            """
            Prune model for each parameter in pruning_grid
            """
            print("- Pruning...")
            for prune_thresh in pruning_grid:
                print(f"> Prune = {prune_thresh}")
                pruned_model = prune_model(grid_params, job_id,
                                           JOB_RUNS_FOLDER,
                                           TRAIN_BASELINE_MODELS_JOB_ID,
                                           prune_thresh,
                                           pruning_training_params,
                                           X_train, y_train)

                save_baseline_model(pruned_model, X_test, y_test,
                                    os.path.join(job_folder(TRAIN_BASELINE_MODELS_JOB_ID, job_id), f"pruned_{prune_thresh}"),
                                    grid_params, training_params)

            tracker.mark_as_completed(job_id)
