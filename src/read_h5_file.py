import h5py
import numpy as np
import yaml

filename = r'C:\Users\matti\github\DSNet\datasets\eccv16_dataset_ovp_google_pool5.h5'

def scan_hdf5(path, recursive=True, tab_step=2):
    def scan_node(g, tabs=0):
        print(' ' * tabs, g.name)
        for k, v in g.items():
            if isinstance(v, h5py.Dataset):
                print(' ' * tabs + ' ' * tab_step + ' -', v.name)
            elif isinstance(v, h5py.Group) and recursive:
                scan_node(v, tabs=tabs + tab_step)
    with h5py.File(path, 'r') as f:
        scan_node(f)

print(scan_hdf5(filename))

with h5py.File(filename, "r") as f:
    
    # List all video
    print("\n")
    print("All video keys: %s" % f.keys())

    # For the first video
    idx = list(f.keys())[0]

    # Get the variable name
    variables = list(f[idx])

    print("\nFor each video, there are the following variables:")
    print(variables)
    print("\n")

    #print(f[idx].value)
