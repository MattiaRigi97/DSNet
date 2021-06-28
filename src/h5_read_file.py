import h5py
import numpy as np
import pickle

filename = r'C:\Users\matti\github\DSNet\datasets\eccv16_dataset_summe_google_pool5.h5'
#filename = r'C:\Users\matti\github\DSNet\datasets\eccv16_dataset_tvsum_google_pool5.h5'


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
	#List all video
	print("\n")
	print("All video keys: %s" % f.keys())

	#For the first video
	idx = list(f.keys())[0]

	#Get the variable name
	variables = list(f[idx])

	print("\nFor each video, there are the following variables:")
	print(variables)
	print("\n")

	video_i = "video_1"

	print(f[video_i]["video_name"][()].decode("utf-8"))
	# print(np.asarray(f[video_i]["features"]))
	# print(np.asarray(f[video_i]["picks"]))
	# print(f[video_i]["features"][0][0:20])

	print("******* DEFAULT FEATURES **********")
	print(f[video_i]["features"][:3])
	print(f[video_i]["features"][:].min())
	print(f[video_i]["features"][:].max())
	print(f[video_i]["features"][:].mean())
	print("\n")
	pickle.dump(f[video_i]["features"][:3], open("original.pk", "wb"))
	
	print("******* LENET FEATURES **********")
	print(f[video_i]["seq_lenet"][:3])
	print(f[video_i]["seq_lenet"].shape)
	print(f[video_i]["seq_lenet"][:].min())
	print(f[video_i]["seq_lenet"][:].max())
	print(f[video_i]["seq_lenet"][:].mean())
	print("\n")
	pickle.dump(f[video_i]["seq_lenet"][:3], open("mattialenet.pk", "wb"))

