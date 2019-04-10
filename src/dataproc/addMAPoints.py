import numpy as np
import h5py
import os
import tqdm

PROJ_ROOT = os.path.abspath(os.path.join(os.pardir, os.pardir))
print(PROJ_ROOT)

ma_path = os.path.join(PROJ_ROOT, 'data/processed/ma_labeled')
ma_h5files = [os.path.join(root, name)
              for root, dirs, files in os.walk(ma_path)
              for name in files
              if name.endswith(('.h5'))]

for ma_h5file in tqdm.tqdm(ma_h5files):
    ma_data = h5py.File(ma_h5file)
    points = ma_data['reference'][:]
    left_points = points[:, :2]
    right_points = points[:, -2:]
    points = np.concatenate(
        (left_points[:, None, :], right_points[:, None, :]),
        axis=1)

    file_id = ma_h5file.split('/')[-2] + '/' + ma_h5file.split('/')[-1]

    ds_h5file = os.path.join(PROJ_ROOT,
                             'data/interim/ds_ma_labeled',
                             file_id)

    ds_data = h5py.File(ds_h5file, 'r+')
    ds_data.create_dataset('tissue/ma_points', data=points)

    ds_data.close()
    ma_data.close()
