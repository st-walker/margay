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
NPARTICLES = int(1e3)
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
from margay.models import T20Hor, HDF5FileWithMaybeMPI# , T20HorWithPhaseAdvance

COMM = MPI.COMM_WORLD
N_CORES = COMM.Get_size()
RANK = COMM.Get_rank()

class UnphysicalQuantity(Exception):
    pass

def _delegated_configs(configs):
    return [list(x) for x in np.array_split(list(configs), N_CORES)][RANK]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run T20 beam dynamics simualtions using OCELOT"
            "Example: "
            # "python t20.py --fs 1 100 11 -q 0.2 -d ./ --name test-run --csr"
        )
    )

    parser.add_argument("-n", "--name",
                        default="./t20-default",
                        help="Name of simulation campaign")

    parser.add_argument("-d", "--rootdir",
                        default="./",
                        help="base directory to write output to")

    # tau_group = parser.add_mutually_exclusive_group(required=True)
    # tau_group.add_argument("--fs",
    parser.add_argument("--fs",
                        help="Bunch bunch-lengths in fs to sample: 1,100,10",
                        nargs=3)
    parser.add_argument("--lsn",
                        default=5,
                        type=int,
                        help="Scan peak current with over N values")
    

    parser.add_argument("--csr-scaling",
                        help="CSR scaling as a factor of bunch length",
                        nargs=1,
                        type=float,
                        default=0.1
                        )

    # Just always assume 250pC...
    # parser.add_argument("-q", "--charge",
    #                     help="Bunch charge to sample",
    #                     required=True
    #                     )

    parser.add_argument("-m", "--model", required=False, default="T20FromTL")
    # parser.add_argument("-k", "--kwargs", required=False)
    parser.add_argument("--set",
                        metavar="KEY=VALUE",
                        nargs='+',
                        help="Set a number of key-value pairs "
                        "(do not put spaces before or after the = sign). "
                        "If a value contains spaces, you should define "
                        "it with double quotes: "
                        'foo="this is a sentence". Note that '
                        "values are always treated as strings.")

    parser.add_argument("--csr", action="store_true")
    parser.add_argument("--sc", action="store_true")

    parser.add_argument("--optics", action="store_true")
    parser.add_argument("--twiss-optics", action="store_true")

    # Currently not used actually so comment it out for now...
    # parser.add_argument("-N", "--nparticles", nargs="?",
    #                     default=200000,
    #                     type=int)

    args = parser.parse_args()
    args.csr_scaling = args.csr_scaling

    if args.fs:
        args.fs[0] = float(args.fs[0])
        args.fs[1] = float(args.fs[1])
        args.fs[2] = int(args.fs[2])

    # args.charge = float(args.charge)
    args.kvpairs = parse_vars(args.set)

    for key, value in vars(args).items():
        print(key, value)
    # logger.info(vars(args))

    return args


def parse_var(s):
    """
    Parse a key, value pair, separated by '='
    That's the reverse of ShellArgs.

    On the command line (argparse) a declaration will typically look like:
        foo=hello
    or
        foo="hello world"
    """
    items = s.split('=')
    key = items[0].strip() # we remove blanks around keys, as is logical
    if len(items) > 1:
        # rejoin the rest:
        value = '='.join(items[1:])

    with contextlib.suppress(ValueError):
        value = float(value)

    return (key, value)


def parse_vars(items):
    """
    Parse a series of key-value pairs and return a dictionary
    """
    d = {}

    if items:
        for item in items:
            key, value = parse_var(item)
            d[key] = value
    return d



def write_beam_to_group(group, parray):
    group.create_dataset("rparticles", data=parray.rparticles)
    group.create_dataset("q_array", data=parray.q_array)
    group.create_dataset("E", data=parray.E)
    group.create_dataset("s", data=parray.s)


def save_metadata(filename, conf, machine, imax_ka):
    with h5py.File(filename, "a") as f:
        # Metadata from config
        f.attrs["argv"] = " ".join(sys.argv)
        for name in conf.__annotations__:
            attr = getattr(conf, name)
            if name == "kvpairs":
                f.attrs.update(attr)
            else:
                f.attrs[name] = attr

        f.attrs["bunch-length"] = conf.bunch_length
        f.attrs["duration"] = f.attrs["bunch-length"] / 3e8
        f.attrs["imax"] = imax_ka # kiloamps

        # machine data
        f.attrs["machine-tfs-file"] = machine.table_file
        try:
            t = tfs.read(machine.table_file)
            head = t.headers
        except tfs.TfsFormatError:
            t = pand8.read(machine.table_file)
            head = t.attrs
        
        date = head["DATE"]
        time = head["TIME"]
        title = head["TITLE"]
        f.attrs["machine-timestamp"] = f"{time}---{date}"
        f.attrs["machine-title"] = title



@dataclass
class SimulationConfig:
    name: str

    bunch_length: List # start, stop, num
    rootdir: str = "./"
    csr: bool = True
    sc: bool = False
    optics: bool = False
    csr_scaling: float = 0.1
    lsn: int = 0 # Length scan with N samples
    model: str = "T20FromTL"
    kvpairs: dict = None
    extra_name: str = ""

    @classmethod
    def from_cli(cls):
        args = parse_arguments()
        cls._check_inputs(args)
        bunch_length = args.fs
        if bunch_length is not None:
           bunch_length = [float(args.fs[0]),
                       float(args.fs[1]),
                       int(args.fs[2])]

        return cls(name=args.name,
                   rootdir=args.rootdir,
                   bunch_length=bunch_length,
                   lsn=args.lsn,
                   csr=args.csr,
                   sc=args.sc,
                   optics=args.optics,
                   csr_scaling=args.csr_scaling,
                   model=args.model,
                   kvpairs=args.kvpairs,
                   )

    @staticmethod
    def _check_inputs(args):
        physics = args.csr or args.sc
        if args.optics and physics:
            logger.warning(
                "Optics are being calculated even with physics enabled."
            )

    @classmethod
    def from_file(cls, filename):
        pass

    def to_file(self, filename):
        print("currently not writing config to file")
        pass

    def bunch_lengths(self):
        """get lengths in m"""
        if self.bunch_length:
            return np.linspace(*self.bunch_length) * 1e-15 * 3e8
        elif self.lsn:
            maximum_peak_current = 6000 # 6kA
            minimum_peak_current = 500 # 0.5kA
            sigma_tau_6ka_250pc = 5e-6 # 
            return np.linspace(sigma_tau_6ka_250pc,
                               sigma_tau_6ka_250pc
                               * maximum_peak_current
                               / minimum_peak_current,
                               num=self.lsn)

    def permutations(self):
        bunch_lengths = self.bunch_lengths()
        for bl in bunch_lengths:
        # for duratio<n, charge in itertools.product(bunch_lengths, charges):            
            yield SimulationConfig(name=self.name,
                                   rootdir=self.rootdir,
                                   bunch_length=bl,
                                   csr=self.csr,
                                   sc=self.sc,
                                   optics=self.optics,
                                   csr_scaling=self.csr_scaling,
                                   model=self.model,
                                   kvpairs=self.kvpairs,
                                   extra_name=self.extra_name
                                   )
    def setup_output_dir(self):
        outdir = self.output_directory()
        try:
            os.mkdir(outdir)
        except FileExistsError:
            pass
        self.to_file(outdir / f"config-{self.name}.toml")
        return outdir

    def output_filename(self):
        dt = datetime.datetime.fromtimestamp(time.time())
        timestr = datetime.datetime.strftime(dt, '%m:%d-%H:%M:%S')

        base = (f"{self.model}"
                f"-q={round(BUNCH_CHARGE*1e12, 5)}pC"
                f"-tau={round(self.bunch_length*1e6, 5)}um"
                # f"-{timestr}"
                )
        if self.extra_name:
            base += f"-{self.extra_name}"
        if self.optics:
            base += "-optics"
        return (self.output_directory()
                / f"{base}.hdf5")

    def output_directory(self):
        return Path(self.rootdir) / self.name

def save_beam(filename, array0, array1):
    """Write beam to HDF5 file using metadata conf and input beam array0 and
    output array array1.
    """
    with HDF5FileWithMaybeMPI(filename, "a") as f:
    # with h5py.File(filename, 'a', driver="mpio", comm=MPI.COMM_WORLD) as f:

        group0 = f.create_group("input_distribution")
        group1 = f.create_group("output_distribution")

        write_beam_to_group(group0, array0)
        write_beam_to_group(group1, array1)

def write_all_optics_datasets(group, twiss_df, names):
    for name in twiss_df:
        try:
            data = np.array(twiss_df[name], dtype=float)
        except ValueError: # then it's a string ("id") so fuck it.
            pass
        else:
            group.create_dataset(name, data=data)

def save_optics(outfile, conf, beam_twiss):
    if not conf.optics:
        return
    with HDF5FileWithMaybeMPI(outfile, "a") as f:
        group = f.require_group("/optics")
        optics = ["s", "alpha_x", "beta_x", "alpha_y", "beta_y", "emit_xn",
                  "emit_yn", "emit_x", "emit_y", "Dx", "Dy", "Dxp", "Dyp",
                  "x", "xp", "y", "yp", "tau", "p"]
        write_all_optics_datasets(group, beam_twiss, optics)



def select_model(model_name, *args, **kwargs):
    return getattr(margay.models, model_name)(*args, **kwargs)

def main():
    config = SimulationConfig.from_cli()
    physics = []
    if config.sc:
        physics.append("sc")
    if config.csr:
        physics.append("csr")

    model = select_model(config.model, unit_step=0.01, physics=physics,
                         save_after=["SBend", "Hcor", "Vcor"],
                         **config.kvpairs)

    this_process_permutations = _delegated_configs(config.permutations())
    print(f"Running {len(this_process_permutations)} jobs on core {RANK}")

    outdir = config.setup_output_dir()
    for i, conf in enumerate(this_process_permutations):
        outfile = conf.output_filename()
        if outfile.is_file():
            print(f"Removing existing file: {outfile}")
            logger.debug(f"Removing existing file: {outfile}")
            outfile.unlink()

        array0 = model.make_beam(conf.bunch_length,
                                 BUNCH_CHARGE,
                                 energy=14.,
                                 nparticles=NPARTICLES)

        
        imax_ka = bunch.get_peak_current(array0) / 1e3
        if imax_ka > 10: # Forget > 10 kA
            logger.info(f"Skipping massive imax: {imax_ka}")
            continue
        else:
            logger.info(f"Running simulation with {imax_ka=}")

        beam_twiss, array1 = model.track(deepcopy(array0), outfile, conf)

        # if conf.plot_optics:
        m, t0 = model.make_magnetic_lattice_and_twiss0()
        optics_twiss = twiss(m, tws0=t0, return_df=True)

        # plt.plot(t)
        save_metadata(outfile, conf, model, imax_ka)
        save_optics(outfile, conf, beam_twiss)
        save_beam(outfile, array0, array1)
        logger.info(f"Saved in {outfile}.")

def gaussian_peak_current(bunch_charge, sigma_tau):
    """bunch_charge in C, sigma_tau in m"""
    # bunch length for gaussian at this peak current and bunch charge
    sigma_tau_6ka_250pc = 5e-6
    charge_ratio = bunch_charge / 250e-12

    if abs(bunch_charge) > 1.1e-9:
        raise UnphysicalQuantity(f"Unphysical bunch charge: {bunch_charge=}")

    sigma_tau_ratio = sigma_tau / sigma_tau_6ka_250pc

    return 6 * charge_ratio / sigma_tau_ratio

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
