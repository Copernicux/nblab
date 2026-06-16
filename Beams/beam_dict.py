from .bessel_mat import bessel_mat
from .gaussian import gaussian
from .hermite import hermite
from .parabolic_mat import parabolic_mat
from .sine import sine
from .mathieu_mat import mathieu_mat
from .arbitrary_mat import arbitrary_mat

beams = {
        "bessel" : bessel_mat,
        "gaussian" : gaussian,
        "hermite" : hermite,
        "parabolic" : parabolic_mat,
        "sine" : sine,
        "mathieu" : mathieu_mat,
        "arbitrary" : arbitrary_mat,
        "beam_v1" : arbitrary_mat,
        "beam_v2" : arbitrary_mat,
        "beam_v3" : arbitrary_mat,
        "beam_v4" : arbitrary_mat,
        "beam_v5" : arbitrary_mat,
        "beam_v6" : arbitrary_mat,
        "beam_v7" : arbitrary_mat,
        "beam_v8" : arbitrary_mat,
        "beam_v9" : arbitrary_mat,
        "beam_v10" : arbitrary_mat,
        "planewave" : arbitrary_mat,
        "gaussian_mat" : arbitrary_mat,
                }
