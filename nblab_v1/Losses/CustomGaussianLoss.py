import oldideas.utils as ut 
import tensorflow as tf
import numpy as np 
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss

N = 300
window_size = 10
resolution = 64

def create_cartesian_meshgrid(window_size, resolution):
    x = np.linspace(-window_size, window_size, resolution)
    y = np.linspace(-window_size, window_size, resolution)

    xx, yy = np.meshgrid(x, y)
    return xx.astype("float64"), yy.astype("float64")

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def CustomGaussianLoss(y, y_pred):

    # real_true = y[:,:,:,0]
    # imag_true = y[:,:,:,1]
    len_y_pred = y_pred.shape[1]
    alpha, beta = y_pred[:,:len_y_pred//2], y_pred[:,len_y_pred//2:]
    real_pred, imag_pred = get_pred_beam(alpha, beta)
    int_true = y[:,:,0]
    int_pred = (tf.math.square(real_pred) + tf.math.square(imag_pred))
    loss = K.sum(K.sum(parabolic_mask() * tf.math.abs(int_true - int_pred)))

    return loss

def get_pred_beam(alpha, beta, args):
    window_size = args["window_size"]
    resolution = args["resolution"]
    N = args["problem_size"]
    window_size = args["window_size"]

    xx, yy = ut.create_cartesian_meshgrid(window_size, resolution)
    r, theta = ut.cart2pol(xx, yy)
    phi = 2 * np.pi / N * np.array(list(map(lambda x: x + 1, list(range(N)))))

    z = tf.cast(tf.complex(alpha * tf.math.cos(beta), alpha * tf.math.sin(beta)), tf.complex128)
    
    nb_matrix = tf.stack([tf.complex(tf.math.cos(r * tf.math.cos(phi[n] - theta)), tf.math.sin(r * tf.math.sin(phi[n] - theta))) for n in range(0,N)])
    pred_beam = tf.tensordot(z, nb_matrix, axes = ([1,0]))
    pred_beam = tf.divide(pred_beam, tf.abs(pred_beam))
    return tf.cast(tf.math.real(pred_beam), tf.float32), tf.cast(tf.math.imag(pred_beam), tf.float32)

def gaussian_mask():
    xx, yy = create_cartesian_meshgrid(window_size,resolution)
    r, theta = cart2pol(xx, yy)
    sigma = 10
    zz = np.sqrt(2*np.pi)* np.exp(-(r**2/sigma**2)) / sigma
    zz = tf.constant(zz, dtype=np.float32)
    return zz

def parabolic_mask():
    xx, yy = create_cartesian_meshgrid(window_size,resolution)
    r, theta = cart2pol(xx, yy)
    zz = 1 - (r)**2 / window_size**2
    zz = tf.constant(zz, dtype=np.float32)
    return zz