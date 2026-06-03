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
        "planewave" : arbitrary_mat,
        "gaussian_mat" : arbitrary_mat,
                }
