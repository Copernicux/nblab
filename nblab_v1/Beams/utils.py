import numpy as np 

def create_cartesian_meshgrid(window_size, resolution):
    # Make sure x passes through 0
    # build a numpy array from -window_size to window_size and resolution points that passes through 0

    x = np.linspace(-window_size, window_size, resolution)
    y = np.linspace(-window_size, window_size, resolution)

    if 0 not in x:
        idx = np.searchsorted(x, 0)
        x = np.insert(x, idx, 0)
    
    if 0 not in y:
        idx = np.searchsorted(y, 0)
        y = np.insert(y, idx, 0)
    
    xx, yy = np.meshgrid(x, y)
    return xx, yy

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def get_fft(beam):
    beam = beam[:,:,0] + 1j*beam[:,:,1]
    fft_beam = np.fft.fft2(beam)
    return fft_beam