import h5py
import sys

def main(filename):
    with h5py.File(filename, "r") as f:
        from IPython import embed; embed()

if __name__ == '__main__':
    main(sys.argv[1])

