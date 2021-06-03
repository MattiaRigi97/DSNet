import h5py
import numpy as np

#filename = r'C:\Users\matti\github\DSNet\datasets\eccv16_dataset_summe_google_pool5.h5'
filename = r'C:\Users\matti\github\DSNet\datasets\eccv16_dataset_tvsum_google_pool5.h5'


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

	print(f["video_11"]["video_name"][()].decode("utf-8"))
	# print(np.asarray(f["video_11"]["features"]))
	# print(np.asarray(f["video_11"]["picks"]))
	print(f["video_11"]["features"].shape)
	# print(f["video_11"]["features"][0][0:20])
	print(f["video_11"]["features"][:].min())
	print(f["video_11"]["features"][:].max())
	print(f["video_11"]["features"][:].mean())
	print(f["video_11"]["n_frames"][()])

# RETRIEVE THE video_i | video_name relationship for SUMME Dataset
filename = r'C:\Users\matti\github\DSNet\datasets\eccv16_dataset_summe_google_pool5.h5'
with h5py.File(filename, "r") as f:
    video_names = []
    for i in range(1, len(f) + 1):
        video_name = str("video_") + str(i)
        video_names.append([video_name, f[video_name]["video_name"][()].decode("utf-8")])

# video_names[i] = name = [video_1, "Air_Force_One"]
print(video_names)