import numpy as np
import scipy as sp
import utils as ut 
import matplotlib.pyplot as plt

def hermite(args):
    n_order = args["n_order"]
    m_order = args["m_order"]
    
    window_size, resolution, mu, sigma = args["window_size"], args["resolution"], args["mu"], args["sigma"]

    xx, yy = ut.create_cartesian_meshgrid(window_size, resolution)
    r, theta = ut.cart2pol(xx, yy)

    n_norm = (np.sqrt(2/np.pi))/((2**n_order) * sp.math.factorial(n_order) * sigma)
    m_norm = (np.sqrt(2/np.pi))/((2**m_order) * sp.math.factorial(m_order) * sigma)
    # n_norm, m_norm = 1,1

    hermite_x = n_norm * sp.special.eval_hermite(n_order,np.sqrt(2) * xx) * np.exp(-(r)**2)
    hermite_y = m_norm * sp.special.eval_hermite(m_order,np.sqrt(2) * yy) * np.exp(-(r)**2)
    zz_real = hermite_x * hermite_y * np.cos(-r) 
    zz_imag = hermite_x * hermite_y * np.sin(-r) 
    
    zz = np.array([zz_real, zz_imag]).transpose(1,2,0).astype("float64")
    return zz