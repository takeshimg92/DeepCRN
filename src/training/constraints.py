import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.ops import math_ops


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
        values = tf.random.uniform(shape, minval=-tf.sqrt(3.0 / shape[0]), maxval=-1e-6)  # seed=self.seed
        return values
