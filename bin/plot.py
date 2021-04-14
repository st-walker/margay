import glob
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from ocelot import *
import numpy as np

from margay.files import h5py_group_to_parray
from margay.bunch import get_projected_emittances, get_peak_current
from margay.plot import subplots_with_td20hor

from collections import namedtuple
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


def plot_raw_emittance_growth(files, ax, charge, label, ls):
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

            from IPython import embed; embed()

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

def plot_emittance_growth_versus_s(charge):
    f, (ax,) = subplots_with_td20hor()

    EmittanceInS = namedtuple("EmittanceInS", ["s", "ex1"])

    all_data = []

    for filename in glob.glob("./csr-only/*.hdf5"):
        data = []
        with h5py.File(filename, "r") as f:
            if not np.isclose(charge, f.attrs["charge"]):
                print(f"Skipping file {filename}")
                continue

            duration = f.attrs["duration"]
            markers = f["markers"]
            parray = h5py_group_to_parray(f["input_distribution"])
            s = 0.
            ex0, ey0 = get_projected_emittances(parray)

            data.append(EmittanceInS(s, ex0))

            for name, marker in markers.items():
                s = marker.attrs["s"]
                parray = h5py_group_to_parray(marker)
                ix, iy = get_projected_emittances(parray)

                data.append(EmittanceInS(s, ix))

        data = sorted(data)
        all_data.append((duration, data))

    # from IPython import embed; embed()

    for i, (duration, data) in enumerate(sorted(all_data)):
        if i == 0:
            label=fr"$\sigma_{{\tau}} = {duration:3.0f}$ fs"
        else:
            label=fr"$\sigma_{{\tau}} = {duration:3.0f}$"

        ax.plot([d.s for d in data],
                [d.ex1 for d in data],
                label = label,
                )

    ax.legend(loc="upper right", fontsize="medium")

    ax.set_xlabel("s / m")
    ax.set_ylabel(r"$\varepsilon_{xn}\,/\,\mathrm{mm \cdot mrad}$")

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





def load_dirs_end():
    pass


def plot_markers(filenames):
    pass


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


def main(filename):
    plot_impact_of_sc()
    # plot_emittance_growth_versus_s(charge=0.2)
    # plot_net_growth(filename)

if __name__ == '__main__':
    main(sys.argv[1:])
