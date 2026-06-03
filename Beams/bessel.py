import numpy as np
import scipy as sp
import oldideas.utils as ut 
import matplotlib.pyplot as plt

def bessel(args):
    n_order = args["n_order"]
    window_size, resolution, mu, sigma = args["window_size"], args["resolution"], args["mu"], args["sigma"]

    xx, yy = ut.create_cartesian_meshgrid(window_size, resolution)
    r, theta = ut.cart2pol(xx, yy)
    bessel = sp.special.jn(n_order,r) 
    zz_real = bessel * np.cos(n_order * theta) 
    zz_imag = bessel * np.sin(n_order * theta) 

    bessel_beam = zz_real + zz_imag * 1j
    bessel_phase = np.angle(bessel_beam).astype("float64")
    bessel_intensity = np.abs(bessel_beam)**2
    bessel_intensity /= np.sum(np.sum(bessel_intensity))
    bessel_intensity = bessel_intensity.astype("float64")
    
    retrieved_amplitude = inv_phase_retrieval(bessel_phase, bessel_intensity)
    return (bessel_phase, bessel_intensity, retrieved_amplitude)



def inv_phase_retrieval(phase, intensity):
        # Measure the phase of the field
    field = intensity * np.exp(1j * phase)
    # Initialize the amplitude or intensity of the field
    amplitude = np.ones_like(phase)

    # Set the number of iterations and convergence threshold
    num_iterations = 100
    convergence_threshold = 1e-2

    # Perform the phase retrieval algorithm
    for i in range(num_iterations):
        # Fourier transform the amplitude or intensity to get the phase
        field_fourier = np.fft.fft2(amplitude)
        field_fourier_phase = np.exp(1j * np.angle(field_fourier))

        # Invert the Fourier transform of the phase to get the amplitude or intensity
        field_reconstructed = np.fft.ifft2(field_fourier_phase)

        # Update the amplitude or intensity based on the measured phase
        amplitude = np.abs(field_reconstructed) * np.exp(1j * phase)

        # Check for convergence
        error = np.mean(np.abs(field - amplitude * np.exp(1j * phase)))
        if error < convergence_threshold:
            break
    return amplitude
# # Test the function with some input values
# espectroAngular(1, 256, 0.15, 0.01, [2])
