
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import oldideas.utils as ut

class AENB(tf.keras.Model):

    def __init__(self, args):
        super(AENB, self).__init__()
        self.args = args
        self.latent_dim = self.args["problem_size"]//2
        self.window_size = self.args["window_size"]
        self.resolution = self.args["resolution"]

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.resolution, self.resolution,self.problem_size, 2)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(2 * self.latent_dim),
                tf.keras.layers.Reshape((self.latent_dim, 2),dtype=np.float64)
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,2),dtype=np.float64),
                tf.keras.layers.Activation(self.ActivationNBL)
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def ActivationNBL(self, tensor):
        tf.config.run_functions_eagerly(True)
        alpha = tensor[0][:,0]
        beta = tensor[0][:,1]
        N = alpha.shape[0]
        xx, yy = ut.create_cartesian_meshgrid(self.window_size, self.resolution)
        r, theta = ut.cart2pol(xx, yy)
        r, theta = tf.Variable(r), tf.Variable(theta)
        phi = tf.Variable(2 * np.pi / N * np.array(list(map(lambda x: x + 1, list(range(N))))))
       
        real_beam = tf.Variable(np.zeros((self.resolution, self.resolution)))
        imag_beam = tf.Variable(np.zeros((self.resolution, self.resolution)))

        for i in range(N):
            a,b = tf.cast(alpha[i], tf.float64), tf.cast(beta[i], tf.float64)
            p = tf.cast(phi[i], tf.float64)
            real_beam.assign_add(a * K.cos(r * K.cos(p - theta) + b))
            imag_beam.assign_add(a * K.sin(r * K.cos(p - theta) + b))

        pred_beam = tf.reshape(tf.Variable(([real_beam, imag_beam])),(1, self.resolution, self.resolution, 2))
        print("Pred Beam Shape : ", pred_beam.shape)

        return pred_beam