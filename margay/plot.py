import matplotlib.pyplot as plt
from ocelot.cpbd import magnetic_lattice
from ocelot.cpbd.optics import twiss
from ocelot.gui.accelerator import plot_elems
from .models import TD20Hor

def subplots_with_lattice(lattice,
                          nrows=1,
                          machine_plot_gap=0.001,
                          gridspec_kw=None,
                          subplots_kw=None,
                          **fig_kw):

    # Set all the kwargs to be supplied to plt.subplots
    height_ratios = [1]
    height_ratios.extend((nrows) * [4])
    the_gridspec_kw = {"height_ratios": height_ratios,
                       "hspace": 0.05}
    if gridspec_kw is not None:
        the_gridspec_kw.update(gridspec_kw)
    the_subplots_kw = {"sharex": True,
                       "gridspec_kw": the_gridspec_kw}
    if subplots_kw is not None:
        the_subplots_kw.update(subplots_kw)
    the_subplots_kw.update(fig_kw)

    fig, axes = plt.subplots(nrows + 1, **the_subplots_kw)
    machine_axes, *axes = axes

    machine_axes.set_facecolor('none') # make background transparent to allow scientific notation
    machine_axes.get_xaxis().set_visible(False)
    machine_axes.get_yaxis().set_visible(False)
    machine_axes.spines['top'].set_visible(False)
    machine_axes.spines['bottom'].set_visible(False)
    machine_axes.spines['left'].set_visible(False)
    machine_axes.spines['right'].set_visible(False)

    # try:
    #     iter(lattice)
    # except TypeError:
    #     pass
    # else:
    #     from IPython import embed; embed()
    #     lattice = magnetic_lattice.MagneticLattice(lattice)

    plot_elems(fig, machine_axes, lattice, legend=False)

    return fig, axes

def subplots_with_td20hor(*args, **kwargs):
    td20 = TD20Hor()
    lattice, _ = td20.make_lattice()
    return subplots_with_lattice(lattice)
