from pathlib import Path
import glob

import numpy as np
import pandas as pd
from ocelot import *
from ocelot.cpbd.beam import ParticleArray, Twiss, Beam

from . import bunch

def h5py_group_to_parray(group):
    """
    Load beam file in npz format and return ParticleArray

    :param filename:
    :return:
    """
    p_array = ParticleArray()
    for key in 'E', 'q_array', 'rparticles', 's':
        p_array.__dict__[key] = np.array(group[key])
    return p_array

