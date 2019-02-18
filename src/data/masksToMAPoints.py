import h5py
import numpy as np
import skimage
from scipy import ndimage
import tqdm
import os


def files_that_endswith(path, endswith):
    files = [os.path.join(root, name)
             for root, dirs, files in os.walk(path)
             for name in files
             if name.endswith((endswith))]
    return files


def mask_with_frame_num(frame_files, frame_num):
    mask_file = [f for f in frame_files
                 if f.split('/')[-1].split('_')[1] == str(frame_num + 1)][0]
    return mask_file


PROJECT_ROOT = '/Users/torjushaukom/Documents/' +\
    'Studier/Masteroppgave/Strain_estimation'
data_path = os.path.join(PROJECT_ROOT, 'data/interim')

# Find all relevant files, as well as their ID
h5files = files_that_endswith(data_path, '.h5')

file_ids = [path.split('/')[-1][:-3] for path in h5files]

png_left_paths = files_that_endswith(data_path, 'lgt.png')
png_right_paths = files_that_endswith(data_path, 'rgt.png')
print("Processing files")
for file_id in tqdm.tqdm(file_ids):
    # Select right h5file for file ID
    h5file = [f for f in h5files if file_id in f][0]
    data = h5py.File(h5file, 'r+')

    # Select the right masks for file ID
    png_left_files = [f for f in png_left_paths if file_id in f]
    png_right_files = [f for f in png_right_paths if file_id in f]

    if (data['tissue/data'].shape[2] != len(png_left_files) or
        data['tissue/data'].shape[2] != len(png_right_files)):
        raise RuntimeError('Number of frames in data is not equal to the number of masks')

    point_idc = []
    for frame_num in range(data['tissue/data'].shape[2]):
        left_frame_file = mask_with_frame_num(png_left_files, frame_num)
        left_mask = skimage.io.imread(left_frame_file)
        right_frame_file = mask_with_frame_num(png_right_files, frame_num)
        right_mask = skimage.io.imread(right_frame_file)

        x_correction = (data['tissue/data'].shape[0] - left_mask.shape[1]) / 2
        x_correction = int(round(x_correction))

        left_coord = ndimage.center_of_mass(left_mask)
        left_coord = [int(round(c)) for c in left_coord]
        left_coord[1] += x_correction
        right_coord = ndimage.center_of_mass(right_mask)
        right_coord = [int(round(c)) for c in right_coord]
        right_coord[1] += x_correction

        point_idc.append([left_coord, right_coord])
    point_idc = np.array(point_idc)
    data['tissue/mitral_points'] = point_idc
    data.close()
