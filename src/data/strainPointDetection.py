import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

PROJ_ROOT = os.path.abspath(os.path.join(os.pardir, os.pardir))
data_path = os.path.join(PROJ_ROOT, 'data/processed/ed_2_es_clips')

h5files = glob.glob(os.path.join(data_path, '*/*.h5'))

for h5file in h5files:
    data = h5py.File(h5file)
    video = np.transpose(data['tissue/data'], [2, 1, 0])
    frame = video[0, :, :]
    mitral_points = data['tissue/mitral_points'][0, :, :]
    print(h5file.split('/')[-2:])

    plt.imshow(frame, cmap='Greys_r')
    plt.scatter(mitral_points[0, 1], mitral_points[0, 0], color='red')
    plt.scatter(mitral_points[1, 1], mitral_points[1, 0], color='red')
    plt.show()
    data.close()
