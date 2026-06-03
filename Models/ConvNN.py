import numpy as np 
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import backend as K
from tensorflow.keras import layers

#Instantiate random seed
tf.random.set_seed(420)

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
            loss = self.compiled_loss(X, y_pred)
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
    
        self.cnn = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(self.resolution, self.resolution, 2)),
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=3, strides=(4, 4), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, strides=(3, 3), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(3, 3), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2 * self.problem_size, activation=self.custom_activation),
        ]
    )

    def call(self, input_tensor):
        y = self.cnn(input_tensor)
        return y
    
    def custom_activation(self, x):
        # print(x.shape)
        # if np.random.random() < 0.0:
        #     return 2 * np.pi * K.hard_sigmoid(x)
        # print(K.relu(x).shape)
        # print(x.shape)
        alpha = x[:, :x.shape[1]//2]
        #normalize alpha
        beta = x[:, x.shape[1]//2:]

        alpha = tf.nn.softmax(alpha)
        alpha = alpha / tf.norm(alpha)
        beta = 2 * np.pi * tf.nn.softmax(beta) 
        # print(alpha[:,:alpha.shape[1]//2].shape, beta[:,beta.shape[1]//2:].shape)
        res = tf.concat([alpha,beta], 1)
        # Normalize res
        # res = res / tf.norm(res)
        return res
        # return x