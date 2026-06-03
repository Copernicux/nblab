import numpy as np 
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import initializers

#Instantiate random seed
tf.random.set_seed(420)



class FFTLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(FFTLayer, self).__init__()

    def call(self, input):
        fft = tf.signal.fft2d(tf.cast(input, tf.complex64))
        return tf.abs(fft)
    
    
import tensorflow as tf
from tensorflow import keras

class CustomFit(keras.Model):
    def __init__(self, model, **kwargs):
        # Pasar kwargs es buena práctica en Keras
        super(CustomFit, self).__init__(**kwargs)
        self.model = model 

    def call(self, inputs, training=False):
        # PROPAGAR EL ARGUMENTO TRAINING
        # Es vital pasar 'training' para capas como Dropout o BatchNormalization
        return self.model(inputs, training=training)

    def train_step(self, data):
        # Desempaquetar datos (Keras maneja si es (x,y) o (x,y,sample_weight))
        x, y = data
        
        with tf.GradientTape() as tape:
            # 1. Forward pass (llamando a self para que propague al modelo interno)
            y_pred = self(x, training=True)
            
            # 2. Calcular pérdida base (MSE, etc.)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            
        # 3. Extraer variables entrenables de forma explícita del modelo interno
        training_vars = self.model.trainable_variables
        
        # 4. Calcular gradientes
        gradients = tape.gradient(loss, training_vars)

        # 5. Aplicar gradientes
        self.optimizer.apply_gradients(zip(gradients, training_vars))
        
        # 6. Actualizar métricas
        self.compiled_metrics.update_state(y, y_pred)
        
        # 7. Retornar diccionario de métricas
        return {m.name: m.result() for m in self.metrics}


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
                    filters=4, kernel_size=32, strides=(8, 8), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(
                    filters=4, kernel_size=32, strides=(8, 8), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(
                    filters=4, kernel_size=32, strides=(8, 8), activation='relu', padding='same'),
            ]
        )
        self.fft = FFTLayer()


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
        attended_input = input_tensor * attention_mask

        # Pass the attended input through the multi-head attention network
        x = self.cnn(attended_input)
        # x = self.fft(attended_input)
        x = self.fft(x)
        y = self.mlp(x)

        return y

    def custom_activation(self, x):
        # Split x into two halves, normalize the first half, and then concatenate
        x_1, x_2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:]
        x_1 = x_1 / K.sqrt(K.sum(K.square(x_1), axis=-1, keepdims=True))
        #Randomly x_1 add or subtract a quantity from some elements of x_1
        x = K.concatenate([x_2, x_1], axis=1)
        #Replace first 10 elements of x = 0

        return x