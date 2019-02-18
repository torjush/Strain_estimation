import h5py
import numpy as np
import tqdm
import glob
import os


PROJ_ROOT = os.path.abspath(os.path.join(os.pardir, os.pardir))
print(PROJ_ROOT)

data_path = os.path.join(PROJ_ROOT, 'data/processed/mitral_points_labeled/')
file_paths = glob.glob(os.path.join(data_path, '*/*.h5'))

for file_path in tqdm.tqdm(file_paths):
    h5file = h5py.File(file_path)

    new_file_directory = os.path.join(PROJ_ROOT,
                                      'data/interim/ed_2_es_clips',
                                      file_path.split('/')[-2])
    if not os.path.exists(new_file_directory):
        os.makedirs(new_file_directory)
    new_file_name_base = file_path.split('/')[-1][:-3]

    full_ed_idc = np.where(h5file['tissue/ds_labels'][:] == 1.0)[0]
    ed_idc = []
    es_idc = []

    for ed_idx in full_ed_idc:
        remaining_es = np.where(h5file['tissue/ds_labels'][ed_idx:] == 2.0)[0]
        if remaining_es.size > 0:
            next_es = remaining_es[0] + ed_idx
            es_idc.append(next_es)
            ed_idc.append(ed_idx)
        else:
            break
    for i, _ in enumerate(ed_idc):
        new_file = h5py.File(
            os.path.join(new_file_directory,
                         new_file_name_base + f'_{i}.h5'), 'w')
        ed_idx = ed_idc[i]
        es_idx = es_idc[i]

        ed_time = h5file['tissue/times'][ed_idx]
        es_time = h5file['tissue/times'][es_idx]

        video = h5file['tissue/data'][:]
        clip = video[:, :, ed_idx:es_idx + 1]
        clip_times = h5file['tissue/times'][ed_idx:es_idx + 1]

        ma_points = h5file['tissue/mitral_points'][ed_idx:es_idx + 1, :]

        tissue = new_file.create_group('tissue')
        tissue.create_dataset('data', data=clip)
        tissue.create_dataset('times', data=clip_times)
        tissue.create_dataset('dirx', data=h5file['tissue/dirx'])
        tissue.create_dataset('diry', data=h5file['tissue/diry'])
        tissue.create_dataset('pixelsize', data=h5file['tissue/pixelsize'])
        tissue.create_dataset('origin', data=h5file['tissue/origin'])
        tissue.create_dataset('mitral_points', data=ma_points)
        try:
            tvi_ed_idx = np.argmin(np.abs(h5file['TVI/times'] - ed_time))
            tvi_es_idx = np.argmin(np.abs(h5file['TVI/times'] - es_time))

            tvi_video = h5file['TVI/data'][:]
            tvi_clip = tvi_video[:, :, tvi_ed_idx:tvi_es_idx + 1]
            tvi_clip_times = h5file['TVI/times'][ed_idx:es_idx + 1]

            tvi = new_file.create_group('TVI')
            tvi.create_dataset('data', data=tvi_clip)
            tvi.create_dataset('times', data=tvi_clip_times)
            # Directions swapped after transpose
            tvi.create_dataset('dirx', data=h5file['TVI/diry'])
            tvi.create_dataset('diry', data=h5file['TVI/dirx'])
            tvi.create_dataset('pixelsize', data=h5file['TVI/pixelsize'])
            tvi.create_dataset('origin', data=h5file['TVI/origin'])
        except KeyError as e:
            print(f'{new_file_name_base} is missing TVI/times, skip TVI for this file')
            print(e)
        new_file.close()
