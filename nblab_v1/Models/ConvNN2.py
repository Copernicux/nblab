import numpy as np 
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import initializers
import tensorflow_hub as hub

# Instantiate random seed

tf.random.set_seed(420)

def custom_parabola_activation(x):
    peak_value = 0.5  # change this to the value where your parabola peaks
    scale = 0.1  # adjust this to change the decay rate
    return K.exp(-scale * K.square(x - peak_value))

class CustomFit(keras.Model):
    def __init__(self, model):
        super(CustomFit, self).__init__()
        self.model = model 

    def call(self, input_tensor):
        return self.model.call(input_tensor)

    def train_step(self, data):
        X, y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(X, training=True)
            loss = self.compiled_loss(y, y_pred)
        training_vars = self.trainable_variables
        gradients = tape.gradient(loss, training_vars)

        self.optimizer.apply_gradients(zip(gradients, training_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name : m.result() for m in self.metrics}

class ConvNN(keras.Model):
    def __init__(self, args):
        super(ConvNN, self).__init__()
        self.args = args
        self.resolution = self.args["resolution"]
        self.problem_size = self.args["problem_size"]

        # Define the attention network
        self.attention = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(1, kernel_size=1, strides=1, padding='same', activation='sigmoid'),
            tf.keras.layers.Lambda(custom_parabola_activation),
        ])

        self.cnn = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=4, kernel_size=32, strides=(8, 8), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(
                    filters=4, kernel_size=32, strides=(8, 8), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(
                    filters=4, kernel_size=32, strides=(8, 8), activation='relu', padding='same'),
            ]
        )

        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(4000, activation='tanh'),
            tf.keras.layers.Dense(2000, activation='tanh'),
            tf.keras.layers.Dense(1000, activation='tanh'),
            tf.keras.layers.Dense(500, activation='tanh'),
            tf.keras.layers.Dense(300, activation='tanh'),
            tf.keras.layers.Dense(2 * self.problem_size, activation=self.custom_activation)
        ])

    def call(self, input_tensor):
        # Pass the input through the attention network
        attention_mask = self.attention(input_tensor)
        
        # Multiply the input with the attention mask
        attended_input = input_tensor * attention_mask

        x = self.cnn(attended_input)
        y = self.mlp(x)
        return y

    def custom_activation(self, x):
        # Split x into two halves, normalize the first half, and then concatenate
        x_1, x_2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:]
        x_1 = x_1 / K.sqrt(K.sum(K.square(x_1), axis=-1, keepdims=True))
        x = K.concatenate([x_2, x_1], axis=1)
        return x


