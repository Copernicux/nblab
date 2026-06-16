import numpy as np
import utils as ut 
import matplotlib.pyplot as plt

def gaussian(args):
    window_size, resolution, mu, sigma = args["window_size"], args["resolution"], args["mu"], args["sigma"]
    xx, yy = ut.create_cartesian_meshgrid(window_size, resolution)
    zz_real = np.sqrt(2/np.pi) * np.exp(-((xx - mu)**2 + (yy - mu)**2)/(2*sigma**2))
    zz_imag = np.zeros((zz_real.shape[0],zz_real.shape[1])) 
    zz = np.array([zz_real, zz_imag]).transpose(1,2,0).astype("float64")
    return zz

