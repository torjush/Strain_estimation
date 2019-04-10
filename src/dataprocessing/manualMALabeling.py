import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

plt.ion()
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data-path')
args = parser.parse_args()
root_dir = "../../data/processed/ma_labeled"
# fname_read = os.path.join(root_dir, "RawData/DL2019/_D113031/J24BF7O8.h5")

f_read = h5py.File(args.data_path, 'r')

tissue = np.array(f_read['tissue']['data'])

ref_coord = np.empty(shape=(0,4))
imgs = np.empty(shape=(0,tissue.shape[1],tissue.shape[0]))
print(imgs.shape)

coordinates = [(0,0),(0,0),(0,0)]

for i in range(tissue.shape[2]):
    plt.imshow(np.transpose(tissue[:,:,i]), cmap='gray')
    plt.pause(0.01)
    plt.clf()
    print(i)

for i in range(tissue.shape[2]):
    plt.imshow(np.transpose(tissue[:,:,i]), cmap='gray')
    imgs = np.append(imgs, [np.transpose(tissue[:,:,i])], axis=0)

    fig = plt.gcf()
    fig.set_size_inches(10,10)

    coordinates = plt.ginput(n=3, timeout=0, show_clicks=True)
    plt.clf()
    plt.scatter(coordinates[0][0], coordinates[0][1], color='b', marker='*', alpha=0.2)
    plt.scatter(coordinates[1][0], coordinates[1][1], color='b', marker='*', alpha=0.2)

    ref_coord = np.append(ref_coord, np.array([[coordinates[0][0],
                                                coordinates[0][1],
                                                coordinates[1][0],
                                                coordinates[1][1]]]), axis=0)
    print(ref_coord)
patient_num = int(args.data_path.split('/')[-2][-1])
fname_write = os.path.join(root_dir, f"p{patient_num}_4c4.h5")
f_write = h5py.File(fname_write, 'w')

imgs = f_write.create_dataset("images", data=imgs)
ref_coord = f_write.create_dataset("reference", data=ref_coord)

fpath_write = os.path.join(root_dir, args.data_path.split('/')[-2:][0])
if not os.path.exists(fpath_write):
    os.mkdir(fpath_write)
fname_write = os.path.join(fpath_write, args.data_path.split('/')[-2:][1])
f_write = h5py.File(fname_write, 'w')

imgs = f_write.create_dataset("images", data=imgs)
ref_coord = f_write.create_dataset("reference", data=ref_coord)
