from copy import deepcopy
from ocelot.cpbd.beam import (generate_parray, get_current,
                              global_slice_analysis, get_envelope,
                              moments_from_parray, optics_from_moments)
from collections import namedtuple
        
# 1nC:   6ka is sigma_tau of 20um
# 250pC: 6kA is sigma_tau of 5um
# 100pC: 6kA is sigma_tau of 2um


def get_peak_current(parray):
    _, currents = get_current(parray)
    return max(currents)

# def get_projected_emittances(parray):
def get_slice_emittances(parray):
    """return normalised projected emittances"""
    filter_iter=2
    smooth_param = 0.0005
    nparts_in_slice = 5000
    slice_params = global_slice_analysis(deepcopy(parray),
                                         nparts_in_slice=nparts_in_slice,
                                         smooth_param=smooth_param,
                                         filter_iter=filter_iter)

    return slice_params.emitxn * 1e6, slice_params.emityn * 1e6

# def get_projected_emittances(parray, ref_twiss=None):
#     mean, cov = moments_from_parray(parray)
#     from IPython import embed; embed()
#     return optics_from_moments(mean, cov, parray.E)

    
    # tws = get_envelope(parray, tws_i=ref_twiss)
    # return tws.emit_xn * 1e6, tws.emit_yn * 1e6


    

def get_projected_emittances(parray, ref_twiss=None):
    tws = get_envelope(parray, tws_i=ref_twiss)
    return tws.emit_xn, tws.emit_yn

    
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
