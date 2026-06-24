import os
import numpy as np
import scipy.io as sp

nblab_path = os.environ["NBLAB_PATH"]

def get_loaded_beam(args):
    beam_name = args["beam"]
    beam_resolution = args["resolution"]
    beam_file = sp.loadmat(f"{nblab_path}Beams/beam_data/{beam_name}_{beam_resolution}.mat")
    zz_real = beam_file["M256"].real
    zz_imag = beam_file["M256"].imag

    beam = zz_real + zz_imag * 1j
    beam_phase = np.angle(beam).astype("float64")
    beam_intensity = np.abs(beam)**2
    beam_intensity /= np.sum(np.sum(beam_intensity))
    beam_intensity = beam_intensity.astype("float64")
    fourier_beam_intensity = np.fft.fftshift(np.abs(np.fft.fft2(beam_intensity)))
    beam = np.array([beam.real + beam.imag * 1j]).reshape(args['resolution'], args['resolution'], 1)

    return (beam_intensity, beam_intensity, beam)


def mathieu_mat(args):
    return get_loaded_beam(args)