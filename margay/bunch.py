from copy import deepcopy
from ocelot.cpbd.beam import generate_parray, get_current, global_slice_analysis
from collections import namedtuple
        

def get_peak_current(parray):
    _, currents = get_current(parray)
    return max(currents)

def get_projected_emittances(parray):
    """return normalised projected emittances"""
    filter_iter=2
    smooth_param = 0.0005
    nparts_in_slice = 5000
    slice_params = global_slice_analysis(deepcopy(parray),
                                         nparts_in_slice=nparts_in_slice,
                                         smooth_param=smooth_param,
                                         filter_iter=filter_iter)

    return slice_params.emitxn * 1e6, slice_params.emityn * 1e6

def plot_ps_x(ax, parray):
    pass

def plot_ps_y(ax, parray):
    pass

def plot_ps_z(ax, parray):
    pass

def plot_slice_params(slice_params):
    pass


def get_slice_params(parray):
    nparts_in_slice = 5000
    smooth_param = 0.05
    filter_base = 2
    filter_iter = 2
    slice_params = global_slice_analysis(p_array_copy,
                                         nparts_in_slice,
                                         smooth_param,
                                         filter_base,
                                         filter_iter)
    return slice_params

def show_e_beam(p_array,
                nbins_x=200,
                nbins_y=200,
                interpolation="bilinear",
                inverse_tau=False,
                show_moments=False,
                nfig=40,
                title=None,
                figsize=None,
                grid=True,
                filename=None,
                headtail=True,
                tau_units="mm"
                ):

    p_array_copy = deepcopy(p_array)

    if inverse_tau:
        p_array_copy.tau()[:] *= -1

    fig = plt.figure(nfig, figsize=figsize)
    if title is not None:
        fig.suptitle(title)
    ax_sp = plt.subplot(325)
    plt.title("Energy spread")
    plt.plot(slice_params.s * tau_factor, slice_params.se * 1e-3, "b")
    # plt.legend()
    plt.xlabel(tau_label)
    plt.ylabel("$\sigma_E$ [keV]")
    plt.grid(grid)

    ax_em = plt.subplot(323, sharex=ax_sp)
    plt.title("Emittances")
    emitxn_mm_mrad = np.round(slice_params.emitxn * 1e6, 2)
    emityn_mm_mrad = np.round(slice_params.emityn * 1e6, 2)
    plt.plot(slice_params.s * tau_factor, slice_params.ex, "r",
             label=r"$\varepsilon_x^{proj} = $" + str(emitxn_mm_mrad))
    plt.plot(slice_params.s * tau_factor, slice_params.ey, "b",
             label=r"$\varepsilon_y^{proj} = $" + str(emityn_mm_mrad))
    plt.legend()
    plt.setp(ax_em.get_xticklabels(), visible=False)
    plt.ylabel(r"$\varepsilon_{x,y}$ [$\mu m \cdot rad$]")
    plt.grid(grid)

    ax_c = plt.subplot(321, sharex=ax_sp)
    plt.title("Current")

    if inverse_tau:
        arrow = r"$\Longrightarrow$"
        label = "head " + arrow
        location = "upper right"
    else:
        arrow = r"$\Longleftarrow$"
        label = arrow + " head"
        location = "upper left"

    plt.plot(slice_params.s * tau_factor, slice_params.I, "b")
    # label = r"$I_{max}=$" + str(np.round(np.max(slice_params.I), 1))
    if headtail:
        leg = ax_c.legend([label], handlelength=0, handletextpad=0, fancybox=True, loc=location)
        for item in leg.legendHandles:
            item.set_visible(False)
    plt.setp(ax_c.get_xticklabels(), visible=False)
    plt.ylabel("I [A]")
    plt.grid(grid)

    ax_ys = plt.subplot(326, sharex=ax_sp)

    show_density(p_array_copy.tau() * tau_factor, p_array_copy.y() * 1e3, ax=ax_ys, nbins_x=nbins_x, nbins_y=nbins_y,
                 interpolation=interpolation, xlabel=tau_label, ylabel='y [mm]', nfig=50,
                 title="Side view", figsize=None, grid=grid)
    if show_moments:
        plt.plot(slice_params.s * tau_factor, slice_params.my * 1e3, "k", lw=2)
        plt.plot(slice_params.s * tau_factor, (slice_params.my + slice_params.sig_y) * 1e3, "w", lw=1)
        plt.plot(slice_params.s * tau_factor, (slice_params.my - slice_params.sig_y) * 1e3, "w", lw=1)

    ax_xs = plt.subplot(324, sharex=ax_sp)

    show_density(p_array_copy.tau() * tau_factor, p_array_copy.x() * 1e3, ax=ax_xs, nbins_x=nbins_x, nbins_y=nbins_y,
                 interpolation=interpolation, ylabel='x [mm]',
                 title="Top view", grid=grid, show_xtick_label=False)
    if show_moments:
        plt.plot(slice_params.s * tau_factor, slice_params.mx * 1e3, "k", lw=2)
        plt.plot(slice_params.s * tau_factor, (slice_params.mx + slice_params.sig_x) * 1e3, "w", lw=1)
        plt.plot(slice_params.s * tau_factor, (slice_params.mx - slice_params.sig_x) * 1e3, "w", lw=1)

    ax_ps = plt.subplot(322, sharex=ax_sp)

    show_density(p_array_copy.tau() * tau_factor, p_array_copy.p() * 1e2, ax=ax_ps, nbins_x=nbins_x, nbins_y=nbins_y,
                 interpolation=interpolation, ylabel='$\delta_E$ [%]',
                 title="Longitudinal phase space", grid=grid, show_xtick_label=False)

    if filename is not None:
        plt.savefig(filename)
