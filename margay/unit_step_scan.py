"""Main entrypoint for running TD20 simulations"""
import matplotlib.pyplot as plt


import datetime
import time
import contextlib
import os
import shutil
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import List
import itertools
import argparse
import logging
from ocelot.cpbd.track import twiss
import tfs
import pand8
from ocelot.cpbd.beam import (generate_parray, get_current,
                              global_slice_analysis, get_envelope,
                              moments_from_parray, optics_from_moments)

from ocelot.cpbd.track import ParameterScanner, UnitStepScanner

import numpy as np
np.seterr(all='raise')
from ocelot.cpbd.io import save_particle_array2npz

import logging

logging.basicConfig()#level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# LUXE_BUNCH_LENGTH_RANGE = (30e-6, 50e-6)
# LUXE_BUNCH_DURATION_RANGE = (30e-6 / 3e8, 50e-6 / 3e8)

BUNCH_CHARGE = 250e-12 # 250 picoCoulombs HARDCODED NOW.
NPARTICLES = int(1e5)
# NPARTICLES = int(1e4)

# try:
import h5py
# except ImportError:
#     import importlib.util
#     spec = importlib.util.spec_from_file_location("h5py", "/opt/local/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/h5py/__init__.py")
#     foo = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(foo)


from mpi4py import MPI

from copy import deepcopy

from margay import bunch
import margay.models
from margay.models import T20Hor, HDF5FileWithMaybeMPI, T20FromTL


def select_model(model_name, *args, **kwargs):
    return getattr(margay.models, model_name)(*args, **kwargs)

def main():
    physics=["sc", "csr"
             ]
    model = select_model("T20FromTL", unit_step=0.01, physics=physics,
                         save_after=["SBend", "Hcor", "Vcor"]
                         )

    parray0 = model.make_beam(20e-6,
                             BUNCH_CHARGE,
                             energy=14.,
                             nparticles=NPARTICLES)

    outputfilename = "unit-step-scan.hdf5"
    navi, lattice = model.make_navigator(parray0, outputfilename, physics=physics)
    unit_steps = np.logspace(-2, 5, num=10)
    us_scanner = UnitStepScanner(navi,
                                 unit_steps,
                                 parray0,
                                 outputfilename,
                                 parameter_name="unit_step"
                                 )
    

    us_scanner.scan()

if __name__ == '__main__':
    main()

# def get_central_position_of_twiss

# shot to shot variation = ~4*10-5.

# 70MeV deviation is not true


# energy deviation inside the bunch.  dependent on compression and the bunch shape.
# wakefields add to the chirp after compression.  is the chirp at the experiment.

# chirp is minimised going into the beamline t20.  70MeV / s
# This might be correctable with sextupoles.


# optics varies in the bunch as well.  not correctable with sextupoles.

# beta mismatch ~1.5
