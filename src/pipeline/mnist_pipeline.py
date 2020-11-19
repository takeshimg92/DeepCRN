import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def read_mnist_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    return X_train, X_test, y_train, y_test


def validate_mnist_data(X, y):
    assert np.max(X.flatten()) <= 1, "Data exceeds value of 1. Preprocess again"
    assert np.min(X.flatten()) >= 0, "Data input cannot be smaller than 0. Preprocess again"
    assert np.isnan(X).sum() == 0, "Input X has NAs"
    assert np.isnan(y).sum() == 0, "Input y has NAs"
    return True


def preprocess_mnist_data_training(X, y):
    """
    Normalize inputs and reshape into 1d vector.
    Exports normalizing parameters from training set, and will provide this to test set
    """
    max_X = np.max(X.flatten())
    min_X = np.min(X.flatten())

    new_X = ((X - min_X) / (max_X - min_X)).reshape(X.shape[0], X.shape[1] * X.shape[2])
    new_y = y

    return new_X, new_y, max_X, min_X


def preprocess_mnist_data_test(X, y, max_X, min_X):
    """
    Normalize inputs and reshape into 1d vector
    """
    new_X = ((X - min_X) / (max_X - min_X)).reshape(X.shape[0], X.shape[1] * X.shape[2])
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
        fig, ax = plt.subplots(figsize=(5, 1 * figs_to_plot), nrows=figs_to_plot, ncols=2)
        for i in range(figs_to_plot):
            ax[i][0].imshow(X[i, :, :])
            ax[i][1].imshow(X_new[i, :, :])
        plt.tight_layout()
        plt.show()

    return X_new


def validate_training(X, y):
    validate_mnist_data(X, y)
    return True


def validate_test(X, y):
    validate_mnist_data(X, y)
    return True


def preprocess_mnist_full(X_train_raw, X_test_raw, y_train_raw, y_test_raw, pixel_new_width):
    assert pixel_new_width in range(1, 29), 'Invalid new width'

    X_train = resize_images(X_train_raw, pixel_size=pixel_new_width, plot=False)
    X_test = resize_images(X_test_raw, pixel_size=pixel_new_width, plot=False)

    # subsampled
    X_train, y_train, max_X, min_X = preprocess_mnist_data_training(X_train, y_train_raw)
    X_test, y_test = preprocess_mnist_data_test(X_test, y_test_raw, max_X, min_X)

    # Data validation
    assert validate_training(X_train, y_train)
    assert validate_test(X_test, y_test)

    return X_train, X_test, y_train, y_test
