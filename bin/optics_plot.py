import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt

import numpy as np
import tfs
from ocelot.adaptors.madxx import tfs_to_cell_and_optics
from margay.plot import subplots_with_td20hor

from margay import plot

def plot_madx_vs_ocelot_twiss():
    f, ax = plt.subplots()

def set_s_label(ax):
    ax.set_xlabel(r"$s\,/\,\mathrm{m}$")

OUTDIR = "/Users/Stuart/Repos/margay/bin/"

LINE_STYLES = ['-','--','-.',':']

def beta_plot(beam_optics, tfs_table):
    f, (ax,) = subplots_with_td20hor(figsize=(11,7))
    ls = iter(LINE_STYLES)

    ax.plot(beam_optics["s"],
            beam_optics["beta_x"],
            linestyle=next(ls),
            label=r"Ocelot $x$")
    ax.plot(tfs_table["S"],
            tfs_table["BETX"],
            linestyle=next(ls),
            label="MAD-X $x$")
    ax.plot(beam_optics["s"],
            beam_optics["beta_y"],
            linestyle=next(ls),
            label=r"Ocelot $y$")
    ax.plot(tfs_table["S"],
            tfs_table["BETY"],
            linestyle=next(ls),
            label="MAD-X $y$")
    ax.set_ylabel(r"$\beta\,/\,\mathrm{m}$")
    set_s_label(ax)
    ax.legend(loc="center right")
    # plt.show()
    return f

def emit_plot(beam_optics, tfs_table):
    f, (ax,) = subplots_with_td20hor(figsize=(11,7))
    ax.plot(beam_optics["s"], beam_optics["emit_xn"]*1e6,
            label=r"Ocelot $x$")
    ax.plot(beam_optics["s"], beam_optics["emit_yn"]*1e6,
            label=r"Ocelot $y$")
    ax.set_ylabel(r"$\varepsilon_n\,/\,\mathrm{mm \cdot mrad}$")
    set_s_label(ax)
    ax.legend(loc="center right")
    # plt.show()
    return f

def disp_plot(beam_optics, tfs_table):
    f, (ax,) = subplots_with_td20hor(figsize=(11,7))
    ls = iter(LINE_STYLES)
    ax.plot(beam_optics["s"], beam_optics["Dx"],
            linestyle=next(ls),
            label=r"Ocelot $x$")
    ax.plot(tfs_table["S"],
            tfs_table["DX"],
            linestyle=next(ls),
            label="MAD-X $x$")
    ax.plot(beam_optics["s"], beam_optics["Dy"],
            linestyle=next(ls),
            label=r"Ocelot $y$")
    ax.plot(tfs_table["S"],
            tfs_table["DY"],
            linestyle=next(ls),
            label="MAD-X $y$")
    ax.set_ylabel("$D$ / m")
    set_s_label(ax)
    ax.legend(loc="upper right")
    # plt.show()
    return f

def dispp_plot(beam_optics, tfs_table):
    f, (ax,) = subplots_with_td20hor(figsize=(11,7))
    ls = iter(LINE_STYLES)
    ax.plot(beam_optics["s"], beam_optics["Dxp"],
            linestyle=next(ls),
            label=r"Ocelot $x$")
    ax.plot(tfs_table["S"],
            tfs_table["DPX"],
            linestyle=next(ls),
            label="MAD-X $x$")
    ax.plot(beam_optics["s"], beam_optics["Dyp"],
            linestyle=next(ls),
            label=r"Ocelot $y$")
    ax.plot(tfs_table["S"],
            tfs_table["DPY"],
            linestyle=next(ls),
            label="MAD-X $y$")
    ax.set_ylabel("$D^\prime$ / rad")
    set_s_label(ax)
    ax.legend(loc="upper right")
    # plt.show()
    return f

def xy_plot(beam_optics, tfs_table):
    f, (ax,) = subplots_with_td20hor(figsize=(11,7))
    ls = iter(LINE_STYLES)
    ax.plot(beam_optics["s"], beam_optics["x"]*1e6,
            linestyle=next(ls),
            label=r"Ocelot $x$")
    ax.plot(tfs_table["S"],
            tfs_table["X"]*1e6,
            linestyle=next(ls),
            label="MAD-X $x$")
    ax.plot(beam_optics["s"], beam_optics["y"]*1e6,
            linestyle=next(ls),
            label=r"Ocelot $y$")
    ax.plot(tfs_table["S"],
            tfs_table["Y"]*1e6,
            linestyle=next(ls),
            label="MAD-X $y$")
    ax.set_ylabel(r"$\bar{x},\,\bar{y}\,/\,\mathrm{\mu m}$")
    set_s_label(ax)
    ax.legend(loc="upper right")
    # plt.show()
    return f

def xyp_plot(beam_optics, tfs_table):
    f, (ax,) = subplots_with_td20hor(figsize=(11,7))
    ls = iter(LINE_STYLES)
    ax.plot(beam_optics["s"], beam_optics["xp"]*1e6,
            linestyle=next(ls),
            label=r"Ocelot $x$")
    ax.plot(tfs_table["S"],
            tfs_table["PX"]*1e6,
            linestyle=next(ls),
            label="MAD-X $x$")
    ax.plot(beam_optics["s"], beam_optics["yp"]*1e6,
            linestyle=next(ls),
            label=r"Ocelot $y$")
    ax.plot(tfs_table["S"],
            tfs_table["PY"]*1e6,
            linestyle=next(ls),
            label="MAD-X $y$")
    ax.set_ylabel(r"$\bar{x}^\prime,\,\bar{y}^\prime\,/\,\mathrm{\mu rad}$")
    set_s_label(ax)
    ax.legend(loc="upper right")
    # plt.show()
    return f

def plot_madx_vs_ocelot_beam(filename):

    outdir = Path(filename).parents[0]
    with h5py.File(filename, "r") as f:
        beam_optics = {key: np.array(value)
                       for key, value in f["optics"].items()}

        tfs_path = f.attrs["machine-tfs-file"]
        tfs_table = tfs.read(tfs_path)
        # cell, _ = tfs_to_cell_and_optics(tfs_path)
        # # cell = list(cell)

    # plt.savefig(OUTDIR + 'beta-vs-ocelot.pdf', bbox_inches='tight') 
    beta_plot(beam_optics, tfs_table).savefig(
        str(outdir / "beta-madx-vs-ocelot.pdf"), bbox_inches="tight",
    )
    emit_plot(beam_optics, tfs_table).savefig(
        str(outdir / "emit-ocelot.pdf"), bbox_inches="tight",
    )
    disp_plot(beam_optics, tfs_table).savefig(
        str(outdir / "disp-madx-vs-ocelot.pdf"), bbox_inches="tight",
    )
    dispp_plot(beam_optics, tfs_table).savefig(
        str(outdir / "dispp-madx-vs-ocelot.pdf"), bbox_inches="tight",
    )
    xy_plot(beam_optics, tfs_table).savefig(
        str(outdir / "xy-madx-vs-ocelot.pdf"), bbox_inches="tight",
    )
    xyp_plot(beam_optics, tfs_table).savefig(
        str(outdir / "xpyp-madx-vs-ocelot.pdf"), bbox_inches="tight",
    )
    print(f"WRITTEN TO {outdir}")

def main(filename):
    plot_madx_vs_ocelot_beam(filename)



if __name__ == '__main__':
    main(sys.argv[1])
