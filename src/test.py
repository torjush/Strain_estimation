import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.deformableNet import DeformableNet
from data.dataLoader import DataLoader
from misc.notebookHelpers import ultraSoundAnimation
import argparse
import os
import h5py

tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('-id', '--experiment-id',
                    help='Name of the current experiment',
                    required=True)
parser.add_argument('-l', '--leakage',
                    type=float, default=0.2,
                    help='Leakage parameter in leaky relu units')
parser.add_argument('-ds', '--num-downsampling-layers',
                    type=int, default=1,
                    help='Number of downsampling stages in CNN')
parser.add_argument('-op', '--output-path',
                    default='../output',
                    help='Path to storing/loading output, ' +
                    'such as model weights, logs etc.')
parser.add_argument('-dp', '--data-path',
                    default='../data/processed/strain_point',
                    help='Path to dataset')

args = parser.parse_args()

savedir = os.path.join(args.output_path, 'models',
                       args.experiment_id)

data_loader = DataLoader(args.data_path, shuffle=False)

defnet = DeformableNet(args.num_downsampling_layers, args.leakage)
try:
    defnet.load_weights(os.path.join(savedir, args.experiment_id))
    print(f'Loaded weights from {savedir}')
except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError):
    print('No previous weights found or weights couldn\'t be loaded')
    exit(-1)

fixed, moving = data_loader.loadSample()
with h5py.File(data_loader.h5files[0]) as data:
    left_point = data['tissue/left_points'][0, :]
    tracked_left_points = defnet.trackPoint(fixed, moving, left_point)
    right_point = data['tissue/right_points'][0, :]
    tracked_right_points = defnet.trackPoint(fixed, moving, right_point)

    tracked_points = np.concatenate((tracked_left_points[:, None, :], tracked_right_points[:, None, :]), axis=1)
    video = data['tissue/data'][:]
    fps = 1 / (data['tissue/times'][3] - data['tissue/times'][2])

    anim = ultraSoundAnimation(video, points=tracked_points, fps=fps)
    anim.save('ma_point_track_test.mp4')

