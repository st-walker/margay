import h5py
import sys
from ocelot import *
from ocelot.gui.accelerator import *

def h5py_group_to_parray(group):
    p_array = ParticleArray()
    for key in 'E', 'q_array', 'rparticles', 's':
        p_array.__dict__[key] = np.array(group[key])
    return p_array


def main(filename):
    with h5py.File(filename, "r") as f:
        from IPython import embed; embed()

if __name__ == '__main__':
    main(sys.argv[1])

