import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

data_path = '../../data/raw/Ultrasound_high_fps'

view_and_vals = pd.read_csv(os.path.join(data_path, 'strain_and_view_gt.csv'),
                            delimiter=';')

h5files = glob.glob(os.path.join(data_path, 'p*/*.h5'))
for h5file in h5files:
    file_name = h5file.split('/')[-1][:-3]

    print(file_name)
    view = view_and_vals[view_and_vals['File'] == file_name]['View'].values

    with h5py.File(h5file) as data:
        frame = np.transpose(data['tissue/data'][:, :, 15])
    plt.imshow(frame, cmap='Greys_r')
    plt.title(f'View: {view}')
    plt.show()
