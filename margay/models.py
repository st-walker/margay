import tfs
from ocelot.adaptors import tfs_to_cell_and_optics
from ocelot.cpbd import csr, sc, magnetic_lattice
from ocelot.cpbd.beam import (cov_matrix_from_twiss,
                              cov_matrix_to_parray,
                              get_current,
                              get_envelope,
                              recalculate_ref_particle,
                              moments_from_parray,
                              optics_from_moments
                              )
from ocelot.cpbd.optics import MethodTM, SecondTM, Navigator, twiss
from ocelot.cpbd.track import track
from ocelot.cpbd.physics_proc import PhysProc
from ocelot.cpbd.elements import Marker
import numpy as np
import h5py
from mpi4py import MPI

import os

from . import bunch

class ConfigurationError(Exception): pass

def optics_func(parray, bounds):
    mean, cov = moments_from_parray(parray)
    return optics_from_moments(mean, cov, parray.E)

class HDF5FileWithMaybeMPI:
    def __init__ (self, *args, **kwargs):
        if "OMPI_COMM_WORLD_SIZE" in os.environ:
            kwargs.update({"driver": "mpio", "comm": MPI.COMM_WORLD})
        self.f = h5py.File(*args, **kwargs)

    def __enter__ (self):
        return self.f

    def __exit__ (self, exc_type, exc_value, traceback):
        self.f.close()


class TD20Base:
    def __init__(self, table, unit_step=0.2, physics=None, save_after=None,
                 **config):
        self.tfs = table # tfs.read(table)
        self.table = tfs.read(table)
        self.unit_step = unit_step
        if physics is None:
            physics = []
        self.physics = physics

        if save_after is None:
            save_after = []
        self.save_after = save_after

        self.config = config


    def make_lattice(self):
        cell, optics = tfs_to_cell_and_optics(self.tfs)
        method = MethodTM()
        method.global_method = SecondTM

        new_cell = []
        markers = []
        s = 0
        for i, element in enumerate(cell):
            new_cell.append(element)
            s += element.l
            if type(element).__name__ in self.save_after:
                marker = Marker(f"{element.id}_{i}_marker")
                new_cell.append(marker)
                markers.append((marker, s))

        self.markers = markers
        lattice = magnetic_lattice.MagneticLattice(new_cell, method=method)

        return lattice, optics

    def _make_csr(self, parray):
        sigma_min = 0.1 * np.std(parray.tau())
        return csr.CSR(n_bin=300, m_bin=5, sigma_min=sigma_min)

    def _make_sc(self, parray):

        s = sc.SpaceCharge()
        s.step = 1
        s.nmesh_xyz = [31, 31, 31]

        return s

    def _build_model(self, parray, outputfilename):
        lattice, twiss0 = self.make_lattice()
        navi = Navigator(lattice)
        navi.unit_step = self.unit_step

        for proc in self.physics:
            if proc == "sc":
                navi.add_physics_proc(self._make_sc(parray),
                                      lattice.sequence[0],
                                      lattice.sequence[-1])
            elif proc == "csr":
                navi.add_physics_proc(self._make_csr(parray),
                                      lattice.sequence[0],
                                      lattice.sequence[-1])

        for m, s in self.markers:
            navi.add_physics_proc(BeamToHDF5(m.id, outputfilename, s,
                                             parray),
                                  m, m)


        return lattice, navi

    def track(self, parray, outputfilename, calculate_optics):
        lattice, navi = self._build_model(parray, outputfilename)
        beam_twiss, output_distribution = track(lattice, parray, navi,
                                                calc_tws=calculate_optics,
                                                optics_func=optics_func
                                                )
        return beam_twiss, output_distribution

    def make_beam(self, duration, charge, nparticles=200000):
        """duration in fs, charge in nC."""
        _, twiss0 = self.make_lattice()

        # original_charge = 0.2e-9
        sigma_tau = duration * 1e-15 * 3e8
        # charge = original_charge
        charge *= 1e-9
        # if charge < 1e-11 or charge > 1e-8:
        #     raise ConfigurationError(f"charge is too small or too big {charge}!")
        
        cov = cov_matrix_from_twiss(twiss0.emit_x,
                                    twiss0.emit_y,
                                    sigma_tau=sigma_tau,
                                    sigma_p=1e-4,
                                    # sigma_p=0,
                                    alpha_x=twiss0.alpha_x,
                                    alpha_y=twiss0.alpha_y,
                                    beta_x=twiss0.beta_x,
                                    beta_y=twiss0.beta_y,
                                    dx=twiss0.Dx,
                                    dpx=twiss0.Dxp,
                                    dy=twiss0.Dy,
                                    dpy=twiss0.Dyp
                                    )

        mean = [twiss0.x, twiss0.xp, twiss0.y, twiss0.yp, 0., 0.]
        parray =  cov_matrix_to_parray(mean,
                                    cov,
                                    energy=twiss0.E,
                                    charge=charge,
                                    nparticles=nparticles)

        from margay.bunch import get_peak_current
        # print("peak current = ", get_peak_current(parray)/1e3)

        return parray

    def twiss_optics(self):
        lattice, twiss0 = self.make_lattice()
        return twiss(lattice, tws0=twiss0)


def write_beam_to_group(group, parray):
    group.create_dataset("rparticles", data=parray.rparticles)
    group.create_dataset("q_array", data=parray.q_array)
    group.create_dataset("E", data=parray.E)
    group.create_dataset("s", data=parray.s)

# TODO FOR FUN (?) class WriteEmittances(PhysProc):
#     def __init__(self, id, filename, s):
#         PhysProc.__init__(self)
#         self.id = id
#         self.energy = None
#         self.filename = filename
#         self.s = s
#         self.written = False

    


class BeamToHDF5(PhysProc):
    def __init__(self, id, filename, s, parray):
        PhysProc.__init__(self)
        self.id = id
        self.energy = None
        self.filename = filename
        self.s = s
        self.written = False

        self._preconfigure_output_file(parray)
        # from IPython import embed; embed()

        # self.shape = parray.rparticles.shapet
        # self.dtype = parray.rparticles.dtype

    saved_fields = ["rparticles", "q_array", "E", "s"]

    def _preconfigure_output_file(self, parray):
        # tbh not sure this is really necessary to be in its own bit.  oh well.
        with HDF5FileWithMaybeMPI(self.filename, "a") as f:
            
        # with h5py.File(self.filename, "a", driver="mpio", comm=MPI.COMM_WORLD) as f:
            group = f.require_group(self._group_name())                
            for name in self.saved_fields:
                data = getattr(parray, name)
                try:
                    dtype = data.dtype
                except AttributeError as e:
                    if isinstance(data, float):
                        dtype = np.float64
                        shape = ()
                    else:
                        raise e
                else:
                    shape = data.shape

                group.create_dataset(name, dtype=dtype, shape=shape)
                group.attrs["id"] = self.id
                group.attrs["s"] = self.s

    def _group_name(self):
        return f"markers/{self.id}"

    def apply(self, parray, dz):
        with HDF5FileWithMaybeMPI(self.filename, "a") as f:        
            group = f[self._group_name()]
            for name in self.saved_fields:
                dset = group[name]
                dset[()] = getattr(parray, name)


class TD20(TD20Base):
    def __init__(self):
        pass


class TD20Hor(TD20Base):
    def __init__(self, *args, **kwargs):
        tfsf = "/Users/stuartwalker/physics/optics-td20/twiss_t20m_hor.tfs"
        super().__init__(tfsf, *args, **kwargs)
