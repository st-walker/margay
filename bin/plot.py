import glob
import sys
from pathlib import Path

from collections.abc import MutableSequence
from dataclasses import dataclass
import h5py
import matplotlib.pyplot as plt
from ocelot import *
import numpy as np

from margay.files import h5py_group_to_parray
from margay.bunch import get_projected_emittances, get_peak_current
from margay.plot import subplots_with_tlt20, subplots_with_model
import margay.models
from margay.models import T20FromTL

from contextlib import suppress
from collections import namedtuple
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LINE_STYLES = ["-", "--", "-.", ":", "."]

def plot_impact_of_sc():
    # For a given bunch charge, scan bunch lengthd. 4 lines, sc, no sc, both,

    no_physics = glob.glob("/Users/stuartwalker/repos/margay/bin/no-physics/*.hdf5")
    csr = glob.glob("/Users/stuartwalker/repos/margay/bin/csr-only/*.hdf5")
    sc = glob.glob("/Users/stuartwalker/repos/margay/bin/sc-only/*.hdf5")
    csr_and_sc = glob.glob(
        "/Users/stuartwalker/repos/margay/bin/csr-and-sc/*.hdf5"
    )

    # RAW EMITTANCE AT END
    f, ax = plt.subplots(figsize=(11,7))
    ls = iter(LINE_STYLES)

    # no_physics = glob.glob("/Users/stuartwalker/repos/margay/bin/optics/*.hdf5")
    # mean_no_physics = plot_raw_emittance_growth(no_physics, ax, 0.2, "No physics", next(ls))
    mean_csr = plot_raw_emittance_growth(csr, ax, 0.2, "CSR only", next(ls))
    mean_sc = plot_raw_emittance_growth(sc, ax, 0.2, "SC only", next(ls))
    mean_csr_and_sc = plot_raw_emittance_growth(csr_and_sc, ax, 0.2, "CSR & SC",
                                                next(ls))

    ax.set_ylabel(r"$\varepsilon_x\,/\,\mathrm{mm\cdot mrad}$")
    ax.set_xlabel("RMS Pulse Duration / fs")

    ax.axhline(mean_sc.x0, label=fr"$\epsilon_x^0$ = {mean_sc.x0:.2f} $\mathrm{{mm\cdot mrad}}$",
               color="red",
               linestyle=next(ls))

    ax.legend(loc="upper right")
    f.savefig("./raw-emittance-growth-sc-vs-csr.pdf", bbox_inches="tight")


    # FRACTIONAL INCREASE IN EMITTANCE
    f, ax = plt.subplots(figsize=(11,7))
    ls = iter(LINE_STYLES)

    # no_physics = glob.glob("/Users/stuartwalker/repos/margay/bin/optics/*.hdf5")
    # mean_no_physics = plot_relative_emittance_growth(no_physics, ax, 0.2, "No physics", next(ls))
    mean_csr = plot_relative_emittance_growth(csr, ax, 0.2, "CSR only", next(ls))
    mean_sc = plot_relative_emittance_growth(sc, ax, 0.2, "SC only", next(ls))
    mean_csr_and_sc = plot_relative_emittance_growth(csr_and_sc, ax, 0.2, "CSR & SC",
                                                next(ls))

    ax.set_ylabel(r"$\varepsilon_x^1\,/\,\varepsilon_x^0$")
    ax.set_xlabel("RMS Pulse Duration / fs")

    # ax.axhline(mean_sc.x0, label=fr"$\epsilon_x^0$ = {mean_sc.x0:.2f} $\mathrm{{mm\cdot mrad}}$",
    #            color="red",
    #            linestyle=next(ls))

    ax.legend(loc="upper right")
    # FRACTIONAL INCREASE IN EMITTANCE

    f.savefig("./relative-emittance-growth-sc-vs-csr.pdf", bbox_inches="tight")

    # plt.show()


class EmittanceCollection(list):
    def get(self, name):
        result = [getattr(x, name) for x in self]
        if not result:
            raise ValueError(f"Empty selection for {name}.")
        return result

@dataclass
class EmittanceData:
    duration: float
    x0: float
    y0: float
    x1: float
    y1: float
    xscale: float
    yscale: float

EmittanceData = namedtuple(
    "EmittanceData",
    ["duration", "x0", "y0", "x1", "y1", "xscale", "yscale"]
)


def plot_raw_emittance_growth(files, ax, charge, label, ls):

    data = EmittanceCollection()

    for filename in files:
        with h5py.File(filename, "r") as f:
            if not np.isclose(charge, f.attrs["charge"]):
                print(f"Skipping file {filename}")
                continue

            # from IPython import embed; embed()

            inp = h5py_group_to_parray(f["input_distribution"])
            oup = h5py_group_to_parray(f["output_distribution"])

            duration = f.attrs["duration"]
            ix, iy = get_projected_emittances(inp)
            ox, oy = get_projected_emittances(oup)

            data.append(EmittanceData(duration, ix, iy, ox, oy, ox/ix, oy/iy))
    data = sorted(data)

    ax.plot([d.duration for d in data],
            [d.x1 for d in data],
            label=label,
            linestyle=ls,
            marker="x"
            )

    return EmittanceData(np.mean([x.duration for x in data]),
                         np.mean([x.x0 for x in data]),
                         np.mean([x.y0 for x in data]),
                         np.mean([x.x1 for x in data]),
                         np.mean([x.y1 for x in data]),
                         np.mean([x.xscale for x in data]),
                         np.mean([x.yscale for x in data])
                         )

def get_files(inpath):
    if not Path(inpath).is_dir():
        return [inpath]

    files = list(filesdir.glob("*.hdf5"))

def get_dirname(files):
    dirs = set([Path(f).parent for f in files])
    if len(dirs) != 1:
        raise RuntimeError("ambiguous output directory")
    return min(dirs)

def plot_emittance_growth_versus_s(inpath, charge):
    EmittanceInS = namedtuple("EmittanceInS", ["s", "ex1", "imax"])

    all_data = []

    files = get_files(inpath)

    filesdir = get_dirname(files)

    if not files:
        raise RuntimeError("No input HDF files!")    

    # Check all models are the same:
    for filename in files:
    # for filename in glob.glob("./csr-only/*.hdf5"):
        model_names = set()
        with h5py.File(filename, "r") as f:
            model_names.add(f.attrs["model"])
    if len(model_names) != 1:
        raise RuntimeError(f"Mixing models!  Models used: {model_names}.")


    model_name = min(model_names)
    logger.info(f"Using optics calculated from {model_name} class.")
    model = getattr(margay.models, model_name)()
    twiss_optics = model.twiss_optics()
    twiss_optics = {twiss.s: twiss for twiss in twiss_optics}


    for filename in files:
    # for filename in glob.glob("./csr-only/*.hdf5"):
        data = []
        with h5py.File(filename, "r") as f:
            # from IPython import embed; embed()
            # if not np.isclose(charge, f.attrs["charge"]):
            #     print(f"Skipping file {filename}")
            #     continue

            # from IPython import embed; embed()
            duration = f.attrs["duration"]
            imax = f.attrs["imax"]
            bunch_length = f.attrs["bunch-length"]

            # duration = bunch_length
            markers = f["markers"]
            parray = h5py_group_to_parray(f["input_distribution"])
            s = 0.

            ex0, ey0 = get_projected_emittances(parray, ref_twiss=twiss_optics[s])
            imax = get_peak_current(parray)

            data.append(EmittanceInS(s, ex0, imax))

            for name, marker in markers.items():
                s = marker.attrs["s"]
                # from IPython import embed; embed()
                ref_twiss = twiss_optics[s]

                parray = h5py_group_to_parray(marker)
                ix, iy = get_projected_emittances(parray, ref_twiss=ref_twiss)
                # from IPython import embed; embed()
                imax = get_peak_current(parray)

                data.append(EmittanceInS(s, ix, imax))
        data = sorted(data)
        all_data.append((duration, data))

    # from IPython import embed; embed()
    fig, (ax,) = subplots_with_model(model, figsize=(15, 7.5))
    ax2 = ax.twinx()
    for i, (duration, data) in enumerate(sorted(all_data)):
        # from IPython import embed; embed()
        if i == 0:
            # label=fr"$\sigma_{{\tau}} = {duration}$ fs"
            label=fr"$\sigma_{{\tau}} = {duration:3.0f}$ fs"
        else:
            # label=fr"$\sigma_{{\tau}} = {duration}$"
            label=fr"$\sigma_{{\tau}} = {duration:3.0f}$"

        ax.plot([d.s for d in data],
                [d.ex1*1e6 for d in data],
                label = label,
                )

        # ax2.plot([d.s for d in data],
        #          [d.imax*1e3 for d in data],
        #          linestyle="--"
        #         )
        # ax2.set_ylabel(r"$I_\mathrm{max}$ / kA")
        
    ax.legend(loc="upper right", fontsize="medium")

    ax.set_xlabel("s / m")
    ax.set_ylabel(r"$\varepsilon_{xn}\,/\,\mathrm{mm\,\,mrad}$")
    # ax.set_ylabel(r"$\varepsilon_{xn}\,/\,\mathrm{mm \cdot mrad}$") # CHANGE ME

    outfile = filesdir / Path(f"emittance-growth-versus-s-{filesdir.name}").with_suffix(".pdf")
    fig.savefig(str(outfile), bbox_inches="tight")
    plt.show()
    # from IPython import embed; embed()



def plot_relative_emittance_growth(files, ax, charge, label, ls):
    EmittanceData = namedtuple(
        "EmittanceData",
        ["duration", "x0", "y0", "x1", "y1", "xscale", "yscale"]
    )

    data = []

    for filename in files:
        with h5py.File(filename, "r") as f:
            if not np.isclose(charge, f.attrs["charge"]):
                print(f"Skipping file {filename}")
                continue

            inp = h5py_group_to_parray(f["input_distribution"])
            oup = h5py_group_to_parray(f["output_distribution"])

            duration = f.attrs["duration"]
            ix, iy = get_projected_emittances(inp)
            ox, oy = get_projected_emittances(oup)

            data.append(EmittanceData(duration, ix, iy, ox, oy, ox/ix, oy/iy))
    data = sorted(data)

    ax.plot([d.duration for d in data],
            [d.xscale for d in data],
            label=label,
            linestyle=ls,
            marker="x"
            )

    return EmittanceData(np.mean([x.duration for x in data]),
                         np.mean([x.x0 for x in data]),
                         np.mean([x.y0 for x in data]),
                         np.mean([x.x1 for x in data]),
                         np.mean([x.y1 for x in data]),
                         np.mean([x.xscale for x in data]),
                         np.mean([x.yscale for x in data]))


def plot_emittance_growth_versus_s_with_mux(dirname, charge):
    fig, (ax,) = subplots_with_tlt20(figsize=(10, 5))

    EmittanceInS = namedtuple("EmittanceInS", ["s", "ex1"])

    all_data = []

    logger.info("Using optics calculated from T20FromTL class.")
    twiss_optics = T20FromTL().twiss_optics()
    twiss_optics = {twiss.s: twiss for twiss in twiss_optics}

    filesdir = Path(dirname)

    emittance_at_the_end = []

    # from IPython import embed; embed()

    for filename in filesdir.glob("*.hdf5"):
    # for filename in glob.glob("./csr-only/*.hdf5"):
        data = []
        with h5py.File(filename, "r") as f:

            if not np.isclose(charge, f.attrs["charge"]):
                print(f"Skipping file {filename}")
                continue

            duration = f.attrs["duration"]
            mux = f.attrs["dmux"]
            if type(mux) is np.ndarray:
                continue
                # from IPython import embed; embed()
            print(mux)

            imax = f.attrs["imax"]
            bunch_length = f.attrs["bunch-length"]

            # duration = bunch_length

            markers = f["markers"]
            parray = h5py_group_to_parray(f["input_distribution"])
            s = 0.

            ex0, ey0 = get_projected_emittances(parray)

            data.append(EmittanceInS(s, ex0))

            for name, marker in markers.items():
                s = marker.attrs["s"]

                ref_twiss = twiss_optics[s]

                parray = h5py_group_to_parray(marker)
                ix, iy = get_projected_emittances(parray, ref_twiss=ref_twiss)

                data.append(EmittanceInS(s, ix))

        data = sorted(data)
        all_data.append((mux, data))

    all_data = sorted(all_data)

    # from IPython import embed; embed()
    # from IPython import embed; embed()
    for i, (mux, data) in enumerate(all_data):
        if i == 0:
            label=fr"$\sigma_{{\tau}} = {mux}$ fs"
            # label=fr"$\sigma_{{\tau}} = {mux:3.0f}$ fs"
        else:
            label=fr"$\sigma_{{\tau}} = {mux}$"
            # label=fr"$\sigma_{{\tau}} = {mux:3.0f}$"

        ax.plot([d.s for d in data],
                [d.ex1 for d in data],
                label = label,
                )

    ax.legend(loc="upper right", fontsize="medium")

    ax.set_xlabel("s / m")
    ax.set_ylabel(r"$\varepsilon_{xn}\,/\,\mathrm{mm\,\,mrad}$")
    # ax.set_ylabel(r"$\varepsilon_{xn}\,/\,\mathrm{mm \cdot mrad}$") # CHANGE ME

    outfile = filesdir / Path(f"emittance-growth-versus-s-{filesdir.name}").with_suffix(".pdf")
    fig.savefig(str(outfile), bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots()

    mux = [first for first, _ in all_data]
    ex = [second[-1].ex1 * 1e6 for _, second in all_data]

    ax.plot(mux, ex)
    ax.set_xlabel(r"$\Delta \mu$")
    ax.set_ylabel(r"$\varepsilon_{xn}\,/\,\mathrm{mm\,\,mrad}$")


    # from IPython import embed; embed()




def load_dirs_end():
    pass


def plot_markers(filenames):
    pass


def plot_csr_scaling():
    dirs = glob.glob("csr-scaling-*")
    dirs = sorted(dirs)

    fig, ax = plt.subplots()


    first = True
    for directory in dirs:
        directory = Path(directory)
        result = EmittanceCollection()
        for path in directory.glob("*.hdf5"):
            try:
                f = h5py.File(path, "r")
                duration = f.attrs["duration"]
                csr_scaling = f.attrs["csr_scaling"]
                inp = h5py_group_to_parray(f["input_distribution"])
                oup = h5py_group_to_parray(f["output_distribution"])
                ix, iy = get_projected_emittances(inp)
                ox, oy = get_projected_emittances(oup)

                result.append(
                    EmittanceData(duration, ix, iy, ox, oy, ox/ix, oy/iy)
                )

            except (BlockingIOError, OSError):
                print(f"Skipping blocked file: {path}")
                continue
            except KeyError as e:
                print(f"Skipping file with missing keys: {path}")
                continue
            finally:
                with suppress(NameError):
                    f.close()
        result.sort()

        scaling = float(directory.name.split("csr-scaling-")[1])

        if first:
            label = rf"${scaling} \times \sigma_\tau = \sigma_\mathrm{{min}}$"
            first = False
        else:
            label = rf"${scaling}$"
        ax.plot(result.get("duration"),
                result.get("x1"),
                marker="x",
                label=label)

        ax.set_xlabel(r"$\sigma_\tau\,/\,\mathrm{fs}$")
        ax.set_ylabel(r"$\varepsilon_x\,/\,\mathrm{mm\,mrad}$")
        # ax.set_ylabel(r"$\varepsilon_x\,/\,\mathrm{mm\cdot mrad}$")


    ax.legend(loc="upper right")
    plt.show()








# def plot_net_growth(filenames):
#     f, ax = plt.subplots()
#     results = []
#     for filename in filenames:
#         with h5py.File(filename, "r") as f:
#             inp = load_h5py_group(f["input_distribution"])
#             oup = load_h5py_group(f["output_distribution"])
#             xi, yi = get_projected_emittances(inp)
#             xo, yo = get_projected_emittances(oup)
#             results.append((xi, yi, f.attrs["duration"], xo, yo))
#             from IPython import embed; embed()
#     return results



# def plot_net_growth(filename):
#     f, ax = plt.subplots()
#     results = []
#     with h5py.File(filename, "r") as f:
#         inp = load_h5py_group(f["input_distribution"])
#         oup = load_h5py_group(f["output_distribution"])
#         xi, yi = get_projected_emittances(inp)
#         xo, yo = get_projected_emittances(oup)
#         from IPython import embed; embed()
#         results.append((xi, yi, f.attrs["duration"], xo, yo))

#     return results



# def main(dirname):
#     files = glob.glob(str(dirname / Path("*.hdf5")))

#     plot_net_growth(files)


def main(dirname):
    # plot_impact_of_sc()
    # plot_emittance_growth_versus_s_with_mux(dirname, charge=0.2)
    plot_emittance_growth_versus_s(dirname, charge=0.2)
    # plot_net_growth(filename)
    # plot_csr_scaling()

if __name__ == '__main__':
    main(sys.argv[1])
