import pandas as pd
import tfs
from ocelot.adaptors.tfs import tfs_to_cell_and_optics
from ocelot.adaptors.mad8 import twiss_to_sequence_with_optics
from ocelot.cpbd import csr, sc, magnetic_lattice
from ocelot.cpbd.elements import RBend, SBend
from ocelot.cpbd.beam import (cov_matrix_from_twiss,
                              cov_matrix_to_parray,
                              get_current,
                              get_envelope,
                              recalculate_ref_particle,
                              moments_from_parray,
                              optics_from_moments,
                              beam_matching
                              )
from ocelot.cpbd.optics import MethodTM, SecondTM, Navigator, twiss
from ocelot.cpbd.track import track
from ocelot.cpbd.physics_proc import PhysProc
from ocelot.cpbd.elements import Marker
from ocelot.common.globals import m_e_GeV
import pand8
from ocelot.cpbd.track import ParameterScanner, UnitStepScanner

import numpy as np
import h5py
# from mpi4py import MPI

import os

from . import bunch

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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


class T20Base:
    def __init__(self, table_file, unit_step=0.2, physics=None, save_after=None,
                 **config):
        # Could be mad8 or madx!
        self.table_file = table_file # tfs.read(table_file)
        self.unit_step = unit_step
        if physics is None:
            physics = []
        self.physics = physics

        if save_after is None:
            save_after = []

        self.save_after = save_after

        self.config = config

    def make_cell_and_twiss0(self):
        """return cell, optics"""
        try:
            return tfs_to_cell_and_optics(self.table_file)
        except tfs.TfsFormatError:
            pass # it is I guess mad8..
        return twiss_to_sequence_with_optics(self.table_file)


    def make_magnetic_lattice_and_twiss0(self):
        cell, optics = self.make_cell_and_twiss0()

        new_cell = []
        markers = []
        s = 0
        sa = self.save_after
        for i, element in enumerate(cell):
            new_cell.append(element)
            s += element.l

            if sa == "all" or type(element).__name__ in sa:
                marker = Marker(f"{element.id}_{i}_marker")
                new_cell.append(marker)
                markers.append((marker, s))

        self.markers = markers
        lattice = magnetic_lattice.MagneticLattice(new_cell,
                                                   method={"global": SecondTM})

        return lattice, optics

    def _make_csr(self, parray, conf):
        sigma_min = conf.csr_scaling * np.std(parray.tau()) # default 0.1*
        c = csr.CSR()
        c.n_bin = 300
        c.m_bin = 5
        c.sigma_min = sigma_min
        c.rk_traj = True
        c.energy = parray.E
        return c

    def _make_sc(self, parray, conf):
        s = sc.SpaceCharge()
        s.step = 1
        s.nmesh_xyz = [31, 31, 31]

        return s

    def add_physics(self, navigator, parray, outputfilename, conf# , csr_markers
                    ):
        for proc in self.physics:
            if proc == "sc":
                navigator.add_physics_proc(self._make_sc(parray, conf),
                                      navigator.lat.sequence[0],
                                      navigator.lat.sequence[-1])
            elif proc == "csr":
                csr = self._make_csr(parray, conf)
                navigator.add_physics_proc(csr,
                                           navigator.lat.sequence[0],
                                           navigator.lat.sequence[-1])
                # navigator

                # for _, marker_pairs in csr_markers.items():
                #     for start, stop in marker_pairs:
                #         csr = self._make_csr(parray, conf)
                #         navigator.add_physics_proc(csr, start, stop)


        for m, s in self.markers:
            navigator.add_physics_proc(BeamToHDF5(m.id, outputfilename, s,
                                             parray),
                                  m, m)


    def make_navigator(self, parray, outputfilename, conf):
        mlattice, twiss0 = self.make_magnetic_lattice_and_twiss0()
        def f(ele):
            return isinstance(ele, (RBend, SBend)) and ele.l > 0.1

        dipole_markers = magnetic_lattice.insert_markers_by_predicate(
            mlattice.sequence,
            f,
        )

        # from IPython import embed; embed()
        navi = Navigator(mlattice)
        navi.unit_step = self.unit_step
        self.add_physics(navi, parray, outputfilename, conf)
        return navi, mlattice

    def unit_step_scan(self, parray, outputfilename, conf):
        calculate_optics = False

        if not self.physics:
            raise ConfigurationError("Must have some physics processes for unit step scan!")

        navi, lattice = self.make_navigator(parray, outputfilename, conf)

        unit_steps = np.logspace(-2, 2, num=5)
        us_scanner = UnitStepScanner(navi,
                                     unit_steps,
                                     parray0=parray
                                     outputfilename,
                                     parameter_name="unit_step"
                                     )

        us_scanner.scan()


        return result

    def track(self, parray, outputfilename, conf):
        calculate_optics = conf.optics
        navi, lattice = self.make_navigator(parray, outputfilename, conf)
        if calculate_optics:
            ofunc = optics_func
        else:
            ofunc = None
        beam_twiss, output_distribution = track(lattice, parray, navi,
                                                calc_tws=calculate_optics,
                                                return_df=True
                                                # optics_func=ofunc
                                                )
        return beam_twiss, output_distribution

    def make_beam(self, bunch_length, charge, energy=None, nparticles=200000):
        """duration in duration in s, charge in C."""
        sigma_tau = bunch_length
        logger.info(
            f"Making beam with charge={charge*1e9}nC, "
            f"length={round(sigma_tau*1e6, 4)}us {nparticles=}"
        )
        _, twiss0 = self.make_magnetic_lattice_and_twiss0()

        if energy is None:
            energy = twiss0.E

        emit_xn = 0.6e-6
        emit_yn = 0.6e-6
        egamma = energy / m_e_GeV
        twiss0.emit_x = emit_xn / egamma
        twiss0.emit_y = emit_yn / egamma
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
                                       energy=energy,
                                       charge=charge,
                                       nparticles=nparticles)

        return parray

    def twiss_optics(self, return_df=True):
        lattice, twiss0 = self.make_magnetic_lattice_and_twiss0()
        # return twiss(lattice, tws0=twiss0, return_df=return_df)
        return twiss(lattice, tws0=twiss0) #, return_df=return_df)


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
        tws = get_envelope(parray)
        # print(f"en: ({tws.emit_xn}, {tws.emit_yn}).  eg: ({tws.emit_x}, {tws.emit_y})")
        with HDF5FileWithMaybeMPI(self.filename, "a") as f:
            group = f[self._group_name()]
            for name in self.saved_fields:
                dset = group[name]
                dset[()] = getattr(parray, name)

def print_it(parray):
    tws = get_envelope(parray)
    print(f"Emittances: ({tws.emit_xn}, {tws.emit_yn})")
    # beam_matching(parray.rparticles,
    #               [-5, 5],
    #               self.x_opt,
    #               self.y_opt,
    #               remove_offsets=True)


class PhaseAdvance(PhysProc):
    def __init__(self, mux=0.0, muy=0.0):
        super().__init__()
        self.mux = mux
        self.muy = muy

    def apply(self, parray, dz):
        return
        tws = get_envelope(parray)
        self.x_opt = [tws.alpha_x, tws.beta_x, self.mux]
        self.y_opt = [tws.alpha_y, tws.beta_y, self.muy]

        print(f"\nEmittances before matching: ({tws.emit_xn}, {tws.emit_yn})")


        beam_matching(parray.rparticles,
                      [-5, 5],
                      self.x_opt,
                      self.y_opt,
                      remove_offsets=True)

        tws = get_envelope(parray)

        print(f"\nEmittances after matching: ({tws.emit_xn}, {tws.emit_yn})")
        # new_twiss =



class T20(T20Base):
    def __init__(self, *args, **kwargs):
        tfsf = "/Users/stuartwalker/physics/optics-t20/twiss_t20.tfs"
        super().__init__(tfsf, *args, **kwargs)


class T20Hor(T20Base):
    def __init__(self, *args, **kwargs):
        tfsf = "/Users/stuartwalker/physics/optics-t20/twiss-t20-with-ff-ff-matched.tfs"
        # tfsf = "/Users/stuartwalker/physics/optics-t20/twiss_t20m_hor.tfs"
        super().__init__(tfsf, *args, **kwargs)


# class T20HorWithPhaseAdvance(T20Hor):
#     def __init__(self, dmux=0.0, dmuy=0.0, *args, **kwargs):
#         self.dmux = dmux
#         self.dmuy = dmuy
#         super().__init__(*args, **kwargs)


#     def make_cell_and_twiss0(self):
#         # return super().make_cell_and_twiss0()

#         cell, twiss0 = super().make_cell_and_twiss0()
#         cell = list(cell)
#         index, *_ = np.where([x.id == "BD.3.T20" for x in cell])[0]

#         marker = Marker(eid="phase_advance")

#         self.table.query("KEYWORD == 'SBEND'")["MUX"].diff()

#         diffs = self.table.query("KEYWORD == 'SBEND'")["MUX"].diff()
#         diffs.name = "DELTA_MUX"

#         s_with_mux = pd.concat([diffs,
#                                 self.table[["MUX", "S", "NAME", "KEYWORD"]
#                                            ].iloc[diffs.index]],axis=1
#                                )



#         # dmux = np.pi * (1 - s_with_mux.query("NAME == 'BD.3.T20'").iloc[0]["DELTA_MUX"])

#         # self.dmux = dmux
#         # self.dmux = 0.0
#         # self.dmuy = 0.0

#         self.marker = marker
#         cell.insert(index, marker)
#         return cell, twiss0

#     def add_physics(self, navigator, parray, outputfilename, conf):
#         super().add_physics(navigator, parray, outputfilename, conf)
#         print(f"dmux = {self.dmux}, dmuy = {self.dmuy}")
#         pa = PhaseAdvance(mux=self.dmux, muy=self.dmuy)
#         navigator.add_physics_proc(pa, self.marker, self.marker)

class T20WithFF(T20Base):
    def __init__(self, *args, **kwargs):
        tfsf = "/Users/stuartwalker/physics/optics-t20/twiss_t20_with_ff.tfs"
        super().__init__(tfsf, *args, **kwargs)

class T20FromTL(T20Base):
    def __init__(self, *args, **kwargs):
        tfsf = "/Users/stuartwalker/physics/s2luxe/luxe/TWISS_CL_T20.txt"
        # RMAT_FILE = "/Users/stuartwalker/physics/luxe-beam-dynamics/luxe/RMAT_CL_T20.txt"
        super().__init__(tfsf, *args, **kwargs)
