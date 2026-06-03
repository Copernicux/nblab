import os
import numpy as np
import scipy.io as sp

nblab_path = os.environ["NBLAB_PATH"]

def get_loaded_beam(args):
    beam_name = args["beam"]
    beam_resolution = args["resolution"]
    beam_file = sp.loadmat(f"{nblab_path}/oldideas/Beams/beam_data/{beam_name}_{beam_resolution}.mat")
    # Check if an array contains imgainary values
    zz_real = beam_file["Uo"].real
    zz_imag = beam_file["Uo"].imag

    beam = zz_real + zz_imag * 1j
    beam_intensity = np.abs(beam)**2
    beam_intensity = beam_intensity.astype("float64")
    beam = np.array([beam.real + beam.imag * 1j]).reshape(args['resolution'], args['resolution'], 1)

    return (beam_intensity, beam_intensity, beam)





def arbitrary_mat(args):
    loaded_beam = get_loaded_beam(args)

    return (loaded_beam[0], loaded_beam[1], loaded_beam[2])