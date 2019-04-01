import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.deformableNet import DeformableNet
from misc.notebookHelpers import ultraSoundAnimation
import argparse
import os
import h5py
import glob

tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('-id', '--experiment-id',
                    help='Name of the current experiment',
                    required=True)
parser.add_argument('-l', '--leakage',
                    type=float, default=0.2,
                    help='Leakage parameter in leaky relu units')
parser.add_argument('-ns', '--num-stages',
                    type=int, default=1,
                    help='Number of chained models')
parser.add_argument('-op', '--output-path',
                    default='../output',
                    help='Path to storing/loading output, ' +
                    'such as model weights, logs etc.')
parser.add_argument('-dp', '--data-path',
                    default='../data/processed/strain_point',
                    help='Path to dataset')

args = parser.parse_args()


def trackPoints(defnets, fixed, moving, points, smoothing=0.):
    num_frames = fixed.shape[0] + 1
    tracked_points = np.zeros((num_frames, points.shape[0], 2))

    displacements = np.zeros((fixed.shape[0],
                              fixed.shape[1],
                              fixed.shape[2], 2))
    for i, defnet in enumerate(defnets):
        warped = defnet(fixed, moving)
        displacements += defnet.interpolated_displacements.numpy()
    yy, xx = np.mgrid[:fixed.numpy().shape[1],
                      :fixed.numpy().shape[2]]

    xx = np.tile(xx[None, :, :], [fixed.shape[0], 1, 1])
    yy = np.tile(yy[None, :, :], [fixed.shape[0], 1, 1])

    grid = np.concatenate((xx[:, None, :, :], yy[:, None, :, :]),
                          axis=1)

    warped_grid = grid + np.transpose(displacements, [0, 3, 1, 2])
    for j in range(points.shape[0]):
        x_coord = np.round(points[j, 0]).astype(int)
        y_coord = np.round(points[j, 1]).astype(int)

        tracked_points[0, j, 0] = x_coord
        tracked_points[0, j, 1] = y_coord
        for frame_num in range(num_frames - 1):
            # Find points in next frame
            next_x_coord = warped_grid[frame_num, 0, y_coord, x_coord]
            next_y_coord = warped_grid[frame_num, 1, y_coord, x_coord]
            next_x_coord = np.clip(next_x_coord, 0, fixed.numpy().shape[2] - 1)
            next_y_coord = np.clip(next_y_coord, 0, fixed.numpy().shape[1] - 1)
            next_x_coord = np.round(next_x_coord).astype(int)
            next_y_coord = np.round(next_y_coord).astype(int)

            # Update current points
            x_coord = next_x_coord
            y_coord = next_y_coord
            tracked_points[frame_num + 1, j, 0] =\
                (1 - smoothing) * x_coord +\
                smoothing * tracked_points[frame_num, j, 0]
            tracked_points[frame_num + 1, j, 1] =\
                (1 - smoothing) * y_coord +\
                smoothing * tracked_points[frame_num, j, 1]

    return tracked_points


def distances(tracked_points):
    left_dist = np.square(tracked_points[:, 0, :] - tracked_points[:, 2, :])
    left_dist = np.sqrt(left_dist[:, 0] + left_dist[:, 1])

    right_dist = np.square(tracked_points[:, 1, :] - tracked_points[:, 3, :])
    right_dist = np.sqrt(right_dist[:, 0] + right_dist[:, 1])

    return left_dist, right_dist


savedir = os.path.join(args.output_path, 'models',
                       args.experiment_id)

defnets = [DeformableNet(i, args.leakage)
           for i in range(args.num_stages, 0, -1)]

for i, defnet in enumerate(defnets):
    try:
        defnet.load_weights(os.path.join(savedir, str(i), args.experiment_id))
        print(f'Loaded weights from savedir')
    except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError):
        print('No previous weights found or weights couldn\'t be loaded')
        exit(-1)
h5file = args.data_path

pal_txt = open('../../pal.txt')
line = pal_txt.readline()[:-1]
pal = np.array([float(val) for val in line.split(',')])

with h5py.File(h5file) as data:
    video = data['tissue/data'][:]
    video = np.array([[[pal[int(round(video[i, j, k]))]
                        for k in range(video.shape[2])]
                       for j in range(video.shape[1])]
                      for i in range(video.shape[0])])
    video /= 255.
    fps = 1 / (data['tissue/times'][3] - data['tissue/times'][2])

    fixed = tf.constant(video[:-1, :, :, None],
                        dtype='float32')
    moving = tf.constant(video[1:, :, :, None],
                         dtype='float32')

    points = []

    def clicks(event):
        point = event.xdata, event.ydata
        points.append(point)
        plt.gca().scatter(event.xdata, event.ydata, color='red')
        plt.gcf().canvas.draw()

    fig, ax = plt.subplots()
    ax.imshow(fixed[0, :, :, 0], cmap='Greys_r')
    fig.canvas.mpl_connect('button_press_event', clicks)
    plt.show()
    plt.close()

    points = np.array(points)
    tracked_points = trackPoints(defnets, fixed, moving, points, smoothing=0.)

    anim = ultraSoundAnimation(video,
                               points=tracked_points, fps=fps)
    anim.save(os.path.join(args.output_path, 'ma_point_track_test.mp4'))

    left_dist, right_dist = distances(tracked_points)
    fig, ax = plt.subplots()
    ax.plot(left_dist)
    ax.plot(right_dist)

    ax.legend(['Left distances', 'Right distances'])
    plt.show()
