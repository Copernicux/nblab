import numpy as np 
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import initializers

#Instantiate random seed
# tf.random.set_seed(420)



class FFTLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(FFTLayer, self).__init__()

    def call(self, input):
        fft = tf.signal.fft2d(tf.cast(input, tf.complex64))
        return tf.abs(fft) ** 2
    
    
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


class NLBModel(keras.Model):
    def __init__(self, args):
        super(NLBModel, self).__init__()
        self.args = args
        self.resolution = self.args["resolution"]
        self.problem_size = self.args["problem_size"]

        # Define the attention network
        self.attention = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(1, kernel_size=1, strides=1, padding='same', activation='sigmoid'),
        ])

        self.cnn = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=4, kernel_size=32, strides=(16, 16), activation='relu', padding='same'),
            ]
        )
        self.fft = FFTLayer()


        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(400, activation='relu'),
            tf.keras.layers.Dense(400, activation='relu'),
            tf.keras.layers.Dense(400, activation='relu'),
            tf.keras.layers.Dense(400, activation='relu'),
            tf.keras.layers.Dense(400, activation='relu'),
            tf.keras.layers.Dense(400, activation='relu'),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(2 * self.problem_size, activation=self.custom_activation)
        ])


    

    def call(self, input_tensor):
        # Pass the input through the attention network
        attention_mask = self.attention(input_tensor)
        attended_input = input_tensor * attention_mask
        
        # Pass the attended input through the multi-head attention network
        x = self.cnn(attended_input)
        x = self.fft(x)
        y = self.mlp(x)

        return y

    def custom_activation(self, x):
        # Split x into two halves, normalize the first half, and then concatenate
        x_1, x_2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:]
        
        # x_1 = x_1 / K.sum(x_1, axis=-1, keepdims=True)
        x_min = K.min(x_1, axis=-1, keepdims=True)
        x_1 = (x_1 - x_min) 

        x_2 *= np.pi
        #Randomly x_1 add or subtract a quantity from some elements of x_1
        x = K.concatenate([x_2, x_1], axis=1)
        # return (x - x_min) / (x_max - x_min)
        return x