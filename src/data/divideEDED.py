import h5py
import numpy as np
import tqdm
import glob
import os


PROJ_ROOT = os.path.abspath(os.path.join(os.pardir, os.pardir))
print(PROJ_ROOT)

data_path = os.path.join(PROJ_ROOT, 'data/processed/ds_ma_labeled/')
file_paths = glob.glob(os.path.join(data_path, '*/*.h5'))

for file_path in tqdm.tqdm(file_paths):
    h5file = h5py.File(file_path)

    new_file_directory = os.path.join(PROJ_ROOT,
                                      'data/interim/ed_2_ed_clips',
                                      file_path.split('/')[-2])

    if not os.path.exists(new_file_directory):
        os.makedirs(new_file_directory)
    new_file_name_base = file_path.split('/')[-1][:-3]

    ed_idc = np.where(h5file['tissue/ds_labels'][:] == 1.0)[0]
    video = np.transpose(h5file['tissue/data'], [2, 1, 0])
    for i in range(ed_idc.shape[0] - 1):
        new_file = h5py.File(os.path.join(new_file_directory,
                                          new_file_name_base +
                                          f'_{i + 1}.h5'), 'w')
        ecg_start = np.argmin(
            np.abs(
                h5file['ecg/ecg_times'] - h5file['tissue/times'][ed_idc[i]]))
        ecg_end = np.argmin(
            np.abs(
                h5file['ecg/ecg_times'] - h5file['tissue/times'][ed_idc[i + 1]])) + 1
        new_file.create_dataset(
            'ecg/ecg_data',
            data=h5file['ecg/ecg_data'][ecg_start:ecg_end])
        new_file.create_dataset(
            'ecg/ecg_times',
            data=h5file['ecg/ecg_times'][ecg_start:ecg_end])

        tissue_start = ed_idc[i]
        tissue_end = ed_idc[i + 1] + 1

        new_file.create_dataset(
            'tissue/data',
            data=video[tissue_start:tissue_end, :, :])
        new_file.create_dataset(
            'tissue/times',
            data=h5file['tissue/times'][tissue_start:tissue_end])
        # Directions inverted because of transpose
        new_file.create_dataset(
            'tissue/dirx', data=h5file['tissue/diry'])
        new_file.create_dataset(
            'tissue/diry', data=h5file['tissue/dirx'])
        new_file.create_dataset(
            'tissue/ds_labels',
            data=h5file['tissue/ds_labels'][tissue_start:tissue_end])
        new_file.create_dataset(
            'tissue/ma_points',
            data=h5file['tissue/ma_points'][tissue_start:tissue_end, :, :])
        new_file.create_dataset(
            'tissue/origin',
            data=h5file['tissue/origin'])
        new_file.create_dataset(
            'tissue/pixelsize', data=h5file['tissue/pixelsize'])

        try:
            tvi_start = np.argmin(
                np.abs(
                    h5file['TVI/times'] - h5file['tissue/times'][ed_idc[i]]))

            tvi_end = np.argmin(
                np.abs(
                    h5file['TVI/times'] - h5file['tissue/times'][ed_idc[i + 1]])) + 1
            TVI = np.transpose(h5file['TVI/data'], [2, 1, 0])
            new_file.create_dataset(
                'TVI/data', data=TVI[tvi_start:tvi_end, :, :])
            new_file.create_dataset(
                'TVI/times', data=h5file['TVI/times'][tvi_start:tvi_end])
            # Directions inverted because of transpose
            new_file.create_dataset(
                'TVI/dirx', data=h5file['TVI/diry'])
            new_file.create_dataset(
                'TVI/diry', data=h5file['TVI/dirx'])
            new_file.create_dataset(
                'TVI/origin', data=h5file['TVI/origin'])
            new_file.create_dataset(
                'TVI/pixelsize', data=h5file['TVI/pixelsize'])

        except KeyError as e:
            print(e)
            print('One video is missing TVI times,' +
                  ' skipping TVI for that sample')

        new_file.close()
    h5file.close()
