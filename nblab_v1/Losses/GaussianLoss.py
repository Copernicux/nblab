import oldideas.utils as ut 
import tensorflow as tf
import numpy as np 
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss

def create_cartesian_meshgrid(window_size, resolution):
    x = np.linspace(-window_size, window_size, resolution)
    y = np.linspace(-window_size, window_size, resolution)

    xx, yy = np.meshgrid(x, y)
    return xx.astype("float64"), yy.astype("float64")

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def GaussianLoss(y_true, y_pred):
    # print(y_true.shape)
    # print(y_pred.shape)
    real_true = y_true[:,:,:,0]
    # print(real_true.shape)
    imag_true = y_true[:,:,:,1]
    real_pred = y_pred[:,:,:,0]
    imag_pred = y_pred[:,:,:,1]

    int_true = (tf.math.square(real_true) + tf.math.square(imag_true))
    int_pred = (tf.math.square(real_pred) + tf.math.square(imag_pred))
    loss = K.mean(K.mean(gaussian_mask() * tf.math.abs(int_true - int_pred)))
    # loss = K.sum(tf.keras.losses.KLDivergence(int_true, int_pred))
    # new_y_true = real_true + 1j*imag_true
    # new_y_pred = real_pred + 1j*imag_pred
    
    # loss = K.sum(K.sum(gaussian_mask() * tf.math.abs(tf.math.abs(new_y_true * tf.math.conj(new_y_true)) - tf.math.abs(new_y_pred* tf.math.conj(new_y_pred))), axis=1),axis=1)
    return loss

    
# class GaussianLoss(Loss):

#     def call(self, y_true, y_pred):
#         print(y_true.shape)
#         print(y_pred.shape)
#         real_true = y_true[:,:,:,0]
#         imag_true = y_true[:,:,:,1]
#         real_pred = y_pred[:,:,:,0]
#         imag_pred = y_pred[:,:,:,1]

#         new_y_true = real_true + 1j*imag_true
#         new_y_pred = real_pred + 1j*imag_pred
        
#         loss = K.sum(K.sum(self.gaussian_mask() * tf.math.abs(tf.math.abs(new_y_true) - tf.math.abs(new_y_pred)), axis=1),axis=1)        

def gaussian_mask():
    xx, yy = create_cartesian_meshgrid(10,256)
    r, theta = cart2pol(xx, yy)
    zz = np.sqrt(2*np.pi)* np.exp(-(r**2))
    zz = tf.constant(zz, dtype=np.float32)
    return zz