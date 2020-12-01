import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from tensorflow.keras import layers
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude

from training.constraints import NonPos, negativeHe
from job_scheduler.jobtracker import job_folder


def build_mnist_model(input_side_pixel: int, layer_weights: list, l2reg: float, dropout_list):
    def constrained_dense_layer(nodes, l2reg, seed=1):
        return layers.Dense(nodes,
                            bias_constraint=NonPos(),  # not allow positive bias
                            activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(l2reg),
                            bias_initializer=negativeHe()  # seed
                            )

    # OBSOLETE
    # bias_initializer=tf.keras.initializers.RandomUniform( # initialize biases negatively
    #                         minval=-1.0, maxval=-0.001, seed=seed

    if isinstance(dropout_list, float):
        dropout_list = [dropout_list] * len(layer_weights)

    assert len(layer_weights) == len(dropout_list), "Layer weights and dropout list must have the same length"

    model = tf.keras.models.Sequential()

    # Input
    model.add(layers.Input(shape=(input_side_pixel * input_side_pixel,)))

    # Add n blocks
    for i, (n_weights, perc_dropout) in enumerate(zip(layer_weights, dropout_list)):
        model.add(constrained_dense_layer(n_weights, l2reg, seed=i))
        model.add(layers.Dropout(perc_dropout))

    # Output: keep a linear dense layer for classification
    # make sure to use from_logits=True in your loss
    model.add(layers.Dense(10),
              bias_constraint=NonPos(),
              bias_initializer=negativeHe())

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
            'test_set_accuracy': test_set_accuracy
        }

    weights_dict = {
        'trainable_weights': float(count_trainable_weights(model)),
        'significative_weights': float(count_significative_weights(model))
    }

    print(f"  > Test set acc {round(test_set_accuracy, 3)}")
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


def plot_training_curves(history, metrics: list):
    n_plots = len(metrics)
    fig, ax = plt.subplots(nrows=1, ncols=n_plots, figsize=(5 * n_plots, 4))
    for axx, metric in zip(ax, metrics):
        axx.plot(history.history[metric], label='Train')
        axx.plot(history.history[f'val_{metric}'], label='Test')
        axx.set_title(metric)
        axx.legend()
    plt.show()


def expected_time_left(time_elapsed, current_iter, total_iters, units='mins'):
    # Find time per iteraction
    assert current_iter > 0, 'Current iter must start from 1 and never be zero'
    avg_time = time_elapsed / current_iter

    # expected time left
    expec = avg_time * (total_iters - current_iter)

    print(
        f"({current_iter}/{total_iters}) Elapsed: {round(time_elapsed)},   Expected time left: {round(expec)} {units}")


def base_weights_path(job_id, JOB_RUNS_FOLDER, TRAIN_BASELINE_MODELS_JOB_ID):
    return os.path.join(JOB_RUNS_FOLDER, TRAIN_BASELINE_MODELS_JOB_ID, job_id, 'weights')


def prune_model(grid_params, job_id, JOB_RUNS_FOLDER, TRAIN_BASELINE_MODELS_JOB_ID, prune_thresh, pruning_training_params, X_train, y_train):
    model_for_pruning = build_mnist_model(**grid_params)
    model_for_pruning.load_weights(base_weights_path(job_id, JOB_RUNS_FOLDER, TRAIN_BASELINE_MODELS_JOB_ID)).expect_partial()

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
                          batch_size=batch_size_prune, epochs=epochs_prune,
                          validation_split=val_split_prune,
                          callbacks=[
                              tfmot.sparsity.keras.UpdatePruningStep(),
                          ])
    return model_for_pruning