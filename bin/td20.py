"""Main entrypoint for running TD20 simulations"""

import os
import shutil
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import List
import itertools
import argparse
import logging

import numpy as np
np.seterr(all='raise') 
from ocelot.cpbd.io import save_particle_array2npz

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
from margay.models import TD20Hor, HDF5FileWithMaybeMPI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

COMM = MPI.COMM_WORLD
N_CORES = COMM.Get_size()
RANK = COMM.Get_rank()

def _delegated_configs(configs):
    return [list(x) for x in np.array_split(list(configs), N_CORES)][RANK]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run TD20 beam dynamics simualtions using OCELOT"
            "Example: "
            "python td20.py --fs 1 100 11 -q 0.2 -d ./ --name test-run --csr"
        )
    )

    parser.add_argument("-n", "--name",
                        required=True,
                        help="Name of simulation campaign")

    parser.add_argument("-d", "--rootdir",
                        required=True,
                        help="base directory to write output to")

    parser.add_argument("--fs",
                        help="Bunch durations in fs to sample: 1,100,10",
                        nargs=3,
                        required=True,
                        )

    parser.add_argument("-q", "--charge",
                        help="Bunch charge to sample",
                        required=True
                        )

    parser.add_argument("--csr", action="store_true")
    parser.add_argument("--sc", action="store_true")

    parser.add_argument("--optics", action="store_true")

    # Currently not used actually so comment it out for now...
    # parser.add_argument("-N", "--nparticles", nargs="?",
    #                     default=200000,
    #                     type=int)

    args = parser.parse_args()

    args.fs[0] = float(args.fs[0])
    args.fs[1] = float(args.fs[1])
    args.fs[2] = int(args.fs[2])

    args.charge = float(args.charge)
    return args


@dataclass
class SimulationConfig:
    name: str
    rootdir: str
    duration: List # start, stop, num
    charge: List # start, stop num
    csr: bool
    sc: bool
    optics: bool

    @classmethod
    def from_cli(cls):
        args = parse_arguments()
        cls._check_inputs(args)
        return cls(name=args.name,
                   rootdir=args.rootdir,
                   duration=[float(args.fs[0]),
                              float(args.fs[1]),
                              int(args.fs[2])],
                   charge=[float(args.charge),
                           float(args.charge),
                           1],
                   csr=args.csr,
                   sc=args.sc,
                   optics=args.optics
                   )

    @staticmethod
    def _check_inputs(args):
        physics = args.csr or args.sc
        if args.optics and physics:
            logger.debug(
                "Optics are being calculated even with physics enabled."
            )

    @classmethod
    def from_file(cls, filename):
        pass

    def to_file(self, filename):
        print("currently not writing myself to file")
        pass

    def permutations(self):
        durations = np.linspace(*self.duration)
        charges = np.linspace(*self.charge)

        for duration, charge in itertools.product(durations, charges):
            yield SimulationConfig(name=self.name,
                                   rootdir=self.rootdir,
                                   duration=duration,
                                   charge=charge,
                                   csr=self.csr,
                                   sc=self.sc,
                                   optics=self.optics,
                                   )

    # @staticmethod
    # def _permutation_name(duration, charge):
    #     fs = duration
    #     q = charge
    #     outputfilename = f"{q=}_{fs=}.npz"
    #     return outputfilename

    def setup_output_dir(self):
        outdir = self.output_directory()
        try:
            os.mkdir(outdir)
        except FileExistsError:
            pass
        self.to_file(outdir / f"config-{self.name}.toml")
        return outdir

    def output_filename(self):
        base = f"nC={self.charge}-fs={self.duration}"
        if self.optics:
            base += "-optics"
        return (self.output_directory()
                / f"{base}.hdf5")

    def output_directory(self):
        return Path(self.rootdir) / self.name



def write_beam_to_group(group, parray):
    group.create_dataset("rparticles", data=parray.rparticles)
    group.create_dataset("q_array", data=parray.q_array)
    group.create_dataset("E", data=parray.E)
    group.create_dataset("s", data=parray.s)


def save_metadata(filename, conf, machine):
    with h5py.File(filename, "a") as f:
        # Metadata from config
        for name in conf.__annotations__:
            f.attrs[name] = getattr(conf, name)
        # machine data
        f.attrs["machine-tfs-file"] = machine.tfs
        header = machine.table.headers
        date = header["DATE"]
        time = header["TIME"]
        f.attrs["machine-timestamp"] = f"{time}---{date}"
        f.attrs["machine-title"] = header["TITLE"]


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

def write_all_optics_datasets(group, twiss_list, names):
    for name in names:
        data = np.array([getattr(x, name) for x in twiss_list])
        group.create_dataset(name, data=data)

def save_optics(outfile, conf, beam_twiss):
    if not conf.optics:
        return
    with HDF5FileWithMaybeMPI(outfile, "a") as f:
    # with h5py.File(outfile, 'a', driver="mpio", comm=MPI.COMM_WORLD) as f:
        group = f.require_group("/optics")
        optics = ["s", "alpha_x", "beta_x", "alpha_y", "beta_y", "emit_xn",
                  "emit_yn", "emit_x", "emit_y", "Dx", "Dy", "Dxp", "Dyp",
                  "x", "xp", "y", "yp", "tau", "p"]
        write_all_optics_datasets(group, beam_twiss, optics)



def main():
    config = SimulationConfig.from_cli()
    physics = []
    if config.sc:
        physics.append("sc")
    if config.csr:
        physics.append("csr")

    td20 = TD20Hor(unit_step=0.2, physics=physics,
                   save_after=["SBend", "Hcor", "Vcor"])

    this_process_permutations = _delegated_configs(config.permutations())
    print(f"Running {len(this_process_permutations)} jobs on core {RANK}")

    outdir = config.setup_output_dir()
    for i, conf in enumerate(this_process_permutations):
        outfile = conf.output_filename()
        if outfile.is_file():
            print(f"Removing existing file: {outfile}")
            logger.debug(f"Removing existing file: {outfile}")
            outfile.unlink()

        array0 = td20.make_beam(conf.duration, conf.charge)
        imax_ka = bunch.get_peak_current(array0) / 1e3
        print(f"Next... {i}")
        if imax_ka > 10: # Forget > 10 kA
            print(f"Skipping massive imax: {imax_ka}")
            continue
        else:
            print(f"Running simulation with {imax_ka=}")

        beam_twiss, array1 = td20.track(deepcopy(array0), outfile, conf.optics)
        save_metadata(outfile, conf, td20)
        save_optics(outfile, conf, beam_twiss)
        save_beam(outfile, array0, array1)

if __name__ == '__main__':
    main()
