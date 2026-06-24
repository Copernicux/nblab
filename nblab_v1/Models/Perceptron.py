import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import backend as K
from tensorflow.keras import layers

class NBLayer(layers.Layer):
    def __init__(self, units, input_dim):
        super(NBLayer, self).__init__()
        self.w = self.add_weight(
            name = "w",
            shape = (input_dim,  2),
            initializer = "random_normal",
            trainable = True,
        )

    def call(self, inputs):
        activation = tf.tensordot(inputs, self.w, axes=([1,5],[0,1]))
        return activation

class Perceptron(keras.Model):
    def __init__(self, args):
        super(Perceptron, self).__init__()
        self.args = args
        self.units = self.args["units"]
        self.nb_layer = NBLayer(self.units, 
                                (self.args["problem_size"]))

    def call(self, input_tensor):
        x = self.nb_layer(input_tensor)
        return x

    