import h5py
import numpy as np

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

	video_i = "video_11"

	print(f[video_i]["video_name"][()].decode("utf-8"))
	# print(np.asarray(f[video_i]["features"]))
	# print(np.asarray(f[video_i]["picks"]))
	# print(f[video_i]["features"][0][0:20])

	print("******* DEFAULT FEATURES **********")
	print(f[video_i]["features"][:])
	print(f[video_i]["features"][:].min())
	print(f[video_i]["features"][:].max())
	print(f[video_i]["features"][:].mean())
	print("\n")
	
	print("******* LENET FEATURES **********")
	print(f[video_i]["seq_lenet"].shape)
	print(f[video_i]["seq_lenet"][:].min())
	print(f[video_i]["seq_lenet"][:].max())
	print(f[video_i]["seq_lenet"][:].mean())
	print("\n")

	print("******* FEATURES **********")
	print(f[video_i]["seq_alexnet"].shape)
	print(f[video_i]["seq_mobilenet"].shape)
	print(f[video_i]["seq_squeeze"].shape)
	print(f[video_i]["seq_resnet"].shape)
	print("\n")

	print("******* CPS **********")
	print(f[video_i]["change_points"].shape)
	print(f[video_i]["cps_lenet"].shape)
	print(f[video_i]["cps_alexnet"].shape)
	print(f[video_i]["cps_mobilenet"].shape)
	print(f[video_i]["cps_squeeze"].shape)
	print(f[video_i]["cps_resnet"].shape)
	print("\n")

	#print(f[video_i]["n_frames"][()])
	#print(f[video_i]["change_points"][:])
	#print(f[video_i]["cps_lenet"][:])
	#print(f[video_i]["cps_alexnet"][:])
	#print(f[video_i]["cps_mobilenet"][:])

# RETRIEVE THE video_i | video_name relationship for SUMME Dataset
filename = r'C:\Users\matti\github\DSNet\datasets\eccv16_dataset_summe_google_pool5.h5'
with h5py.File(filename, "r") as f:
    video_names = []
    for i in range(1, len(f) + 1):
        video_name = str("video_") + str(i)
        video_names.append([video_name, f[video_name]["video_name"][()].decode("utf-8")])

# video_names[i] = name = [video_1, "Air_Force_One"]
print(video_names)