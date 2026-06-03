import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def parabolic(args):

    window_size, resolution, mu, sigma = args["window_size"], args["resolution"], args["mu"], args["sigma"]

    dr = args['dr']
    N = args['resolution']
    a = args['a_param']
    r_anillo = args['r_anillo']
    parabolic_type = args['parabolic_type']

    # Cintura del haz gaussiano
    w0 = 1
    # Longitud de ventana numérica
    L = args['window_size'] / 2
    # Intervalo de muestreo
    delta = 2 * L / N
    # longitud de onda He-Ne
    lam = 1
    # Numero de propagacion
    k = 2 * np.pi / lam
    # Frecuencia critica
    fc = 1 / (2 * delta)
    
    kmax = 2 * np.pi * fc
    deltaKx = 2 * kmax / N
    
    # Vector auxiliar
    aux = np.arange(-N/2, N/2)
    kx = deltaKx * aux
    x = delta * aux

    # Ventana numerica
    [KX, KY] = np.meshgrid(kx, kx)
    [thk, rk] = np.arctan2(KY, KX), np.sqrt(KX**2 + KY**2)
    
    [X, Y] = np.meshgrid(x, x)
    [th, r] = np.arctan2(Y, X), np.sqrt(X**2 + Y**2)

    # Espectro Angular
    C = 1 * (rk < (r_anillo + dr) * kmax) * (rk > r_anillo * kmax)
    
    # Parabolic Beam
    epsilon = 1e-4
    Ae = np.exp(1j*a * np.log(epsilon + np.abs(np.tan(thk/2)))) / \
         (epsilon + 2*np.sqrt(np.pi*np.abs(np.sin(thk)))) * \
         ((KY > 0) | (KY < 0)) #Solucion Par
    Ao = -1j * Ae * (KY >= 0) + Ae * (KY <= 0) #Solucion Impar
    
    # Apertura Anular
    if parabolic_type == 'even':
        A = C * Ae
    elif parabolic_type == 'odd':
        A = C * Ao 
    
    U0 = fftshift(fft2(A)) 
    # Apply a gaussian filter to the beam
    intensity = np.abs(U0)**2
    # Normalize energy
    intensity = intensity / np.sum(intensity)

    phase = np.angle(U0)

    retrieved_amplitude = inv_phase_retrieval(phase, intensity)
    

    return (phase, intensity, retrieved_amplitude)


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
