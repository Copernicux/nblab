import numpy as np
import scipy as sp
import oldideas.utils as ut 
from scipy.special import mathieu_a, mathieu_b
from scipy.special import mathieu_a, mathieu_b, mathieu_cem, mathieu_modcem1
import matplotlib.pyplot as plt

def mathieu(args):
    q,a,k,theta = args["q"], args["a"], args["k"], args["theta"]

    # Grid of points from -window_size to window_size
    window_size, resolution, mu, sigma = args["window_size"], args["resolution"], args["mu"], args["sigma"]
    xx, yy = ut.create_cartesian_meshgrid(window_size, resolution)
    r, phi = ut.cart2pol(xx, yy)

    # Calculate the Mathieu functions.
    a_m = mathieu_a(a, q)
    b_n = mathieu_b(a, q)
    ce_mn = mathieu_cem(a,q,phi)
    mc_mn = mathieu_modcem1(a,q,phi)

    ce_mn = ce_mn[0] + ce_mn[1]*1j
    mc_mn = mc_mn[0] + mc_mn[1]*1j
    
    print(ce_mn)
     # Calculate the Mathieu Gaussian function.
    exp_term = np.exp(-1/(args['window_size'])*(r**2))
    mg = (a_m * b_n * np.sqrt(2*np.pi) / mc_mn) * \
         np.cos(q*xx) * ce_mn * exp_term
    
    print(mg.shape)
    
    intensity = (np.abs(mg)**2).astype("float64")
    intensity /= np.sum(np.sum(intensity))

    phase = (np.angle(mg)).astype("float64")

    # print(intensity)
    print(np.max(intensity), np.min(intensity))
    # Plot Intensity and Phase
    plt.figure()
    plt.imshow(intensity)
    plt.colorbar()
    plt.title("Intensity")
    plt.figure()
    plt.imshow(phase)
    plt.colorbar()
    plt.title("Phase")
    plt.show()


    print(intensity.shape, phase.shape)
    raise ValueError('Debugging')
    return (phase, intensity)