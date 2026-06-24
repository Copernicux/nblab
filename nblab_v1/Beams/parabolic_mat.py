import os
import numpy as np
import scipy.io as sp

nblab_path = os.environ["NBLAB_PATH"]

# def get_loaded_beam(args):
#     beam_name = args["beam"]
#     beam_resolution = args["resolution"]
#     beam_file = sp.loadmat(f"{nblab_path}Beams/beam_data/{beam_name}_{beam_resolution}.mat")

#     zz_real = beam_file[f"Haz_parabolico_{args['resolution']}"].real
#     zz_imag = beam_file[f"Haz_parabolico_{args['resolution']}"].imag

#     beam = zz_real + zz_imag * 1j
#     beam_phase = np.angle(beam).astype("float64")
#     beam_intensity = np.abs(beam)**2
#     beam_intensity /= np.sum(np.sum(beam_intensity))
#     beam_intensity = beam_intensity.astype("float64")
#     fourier_beam_intensity = np.fft.fftshift(np.abs(np.fft.fft2(beam_intensity)))
#     # fourier_beam_intensity = np.abs(np.fft.fft2(beam_intensity))
#     return (beam_intensity, beam_intensity,  beam)

def get_loaded_beam(args):
    beam_name = args["beam"]
    beam_resolution = args["resolution"]
    beam_file = sp.loadmat(f"{nblab_path}Beams/beam_data/{beam_name}_simetrico_{beam_resolution}.mat")

    zz_real = beam_file[f"parabolico_simetrico"].real
    zz_imag = beam_file[f"parabolico_simetrico"].imag


    beam = zz_real + zz_imag * 1j
    beam_phase = np.angle(beam).astype("float64")
    beam_intensity = np.abs(beam)**2
    beam_intensity /= np.sum(np.sum(beam_intensity))
    beam_intensity = beam_intensity.astype("float64")
    beam = np.array([beam.real + beam.imag * 1j]).reshape(args['resolution'], args['resolution'], 1)

    return (beam_intensity, beam_intensity,  beam)



def parabolic_mat(args):
    loaded_beam = get_loaded_beam(args)

    return (loaded_beam[0], loaded_beam[1], loaded_beam[2])