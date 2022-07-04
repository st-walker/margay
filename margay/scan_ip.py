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
from ocelot.cpbd.track import twiss, track
import tfs
import pand8
from ocelot.cpbd.magnetic_lattice import insert_markers_by_name
from ocelot.cpbd.beam import (generate_parray, get_current,
                              global_slice_analysis, get_envelope,
                              moments_from_parray, optics_from_moments)
from ocelot.cpbd.physics_proc import CopyBeam
from ocelot.cpbd import csr, sc, magnetic_lattice

from ocelot.cpbd.track import ParameterScanner, UnitStepScanner

import numpy as np
np.seterr(all='raise')
from ocelot.cpbd.io import save_particle_array2npz
from ocelot import *
import logging

logging.basicConfig()#level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# LUXE_BUNCH_LENGTH_RANGE = (30e-6, 50e-6)
# LUXE_BUNCH_DURATION_RANGE = (30e-6 / 3e8, 50e-6 / 3e8)

BUNCH_CHARGE = 250e-12 # 250 picoCoulombs HARDCODED NOW.
NPARTICLES = int(1e3)
#NPARTICLES = int(1000)

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
from margay.models import T20Hor, HDF5FileWithMaybeMPI, T20FromTL, BeamToHDF5


# class 

N = 8

maximum_peak_current = 6000 # 6kA
minimum_peak_current = 500 # 0.5kA
sigma_tau_6ka_250pc = 5e-6 # 
LENGTHS =  np.linspace(sigma_tau_6ka_250pc,
                       sigma_tau_6ka_250pc
                       * maximum_peak_current
                       / minimum_peak_current,
                       num=10)


def select_model(model_name, *args, **kwargs):
    return getattr(margay.models, model_name)(*args, **kwargs)

# # class BunchScanner(ParameterScanner):
# #     def __init__(self, ):
# #         pass
# class DumpHere(PhysProc):
#     def __init__(self, name=""):
#         PhysProc.__init__(self)
#         self.energy = None
#         self.name = name

#     def apply(self, p_array, dz):
#         self.parray = deepcopy(p_array)

#     def __repr__(self):
#         return f"<Dump: {self.name}, at={hex(id(self))}>"

ENERGY = 14.


class BunchLengthScanner(ParameterScanner):
    def prepare_navigator(self, length, parray, job_index):
        navi = super().prepare_navigator(length, parray, job_index)
        # navi.unit_step = 0.05
        navi.unit_step = 0.01

        # # length = np.std(parray.tau()) # default 0.1*
        # sigma_min = 0.1 * length
        # c = csr.CSR()
        # c.n_bin = 300
        # c.m_bin = 5
        # c.sigma_min = sigma_min
        # c.rk_traj = True
        # c.energy = ENERGY
        
        # from IPython import embed; embed()

        # navi.add_physics_proc(c,
        #                       navi.lat.sequence[0],
        #                       navi.lat.sequence[-1])

        s = sc.SpaceCharge()
        s.step = 1
        s.nmesh_xyz = [31, 31, 31]

        navi.add_physics_proc(s,
                              navi.lat.sequence[0],
                              navi.lat.sequence[-1])

        sa = global_slice_analysis(parray)
        print(f"Peak current pf parray0: {max(sa.I*1e-3)}kA")
        print(f"Projected X emittance of parray0: {sa.emitxn*1e6}um")
        print(f"Projected Y emittance of parray0: {sa.emityn*1e6}um")        


        # navi.add_physics_proc(navi.la
        return navi

def main():
    # physics=["sc", "csr"]
    physics = []
    model = select_model("T20FromTL", unit_step=1., physics=physics)


    parray0s = [model.make_beam(length,
                                BUNCH_CHARGE,
                                energy=14.,
                                nparticles=NPARTICLES,
                                check=True)
                for length in LENGTHS]

    outputfilename = "2-test-peak-current-scan.hdf5"
    navi, lattice = model.make_navigator(parray0s[0],
                                         outputfilename,
                                         physics=physics)

    ip_marker = next(e for e in lattice.sequence if e.id == "IP.LUXE.T20")

    # dump = CopyBeam()
    # navi.add_physics_proc(dump, ip_marker, ip_marker)
    # # from IPython import embed; embed()

    # _, parray1 = track(navi.lat, parray0.copy(), navi=navi)
    # from IPython import embed; embed()
    # unit_steps = np.logspace(-1, 2, num=3)
    # us_scanner = UnitStepScanner(navi,
    #                              unit_steps,
    #                              parray0,
    #                              outputfilename,
    #                              parameter_name="unit_step",
    #                              markers=[ip_marker]
    #                              )

    us_scanner = BunchLengthScanner(navi,
                                    LENGTHS,
                                    parray0s,
                                    parameter_name="bunch_length",
                                    markers=[ip_marker]
                                    )
    
    us_scanner.scan(outputfilename, 
                    nproc=4,
                    run_also_without_physics=True)

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
