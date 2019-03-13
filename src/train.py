import tensorflow as tf
import numpy as np
from models.deformableNet import DeformableNet
from models.losses import nncc2d, bendingPenalty
from data.dataLoader import DataLoader
import matplotlib.pyplot as plt
from misc.plots import plotGrid
import argparse
import os
import io
import h5py
import glob

tf.enable_eager_execution()
parser = argparse.ArgumentParser()
parser.add_argument('-id', '--experiment-id',
                    help='Name of the current experiment',
                    required=True)
parser.add_argument('-s', '--num-steps',
                    type=int, default=10,
                    help='Number of steps to run training')
parser.add_argument('-lr', '--learning-rate',
                    type=float, default=1e-3,
                    help='Learning rate for optimizer')
parser.add_argument('-bp', '--penalty',
                    type=float, default=0.005,
                    help='Impact of the bending penalty in the loss function')
parser.add_argument('-l', '--leakage',
                    type=float, default=0.2,
                    help='Leakage parameter in leaky relu units')
parser.add_argument('-ds', '--num-downsampling-layers',
                    type=int, default=1,
                    help='Number of downsampling stages in CNN')
parser.add_argument('-op', '--output-path',
                    default='../output',
                    help='Path to storing output, ' +
                    'such as model weights, logs etc.')
parser.add_argument('-mp', '--model-path',
                    default='../output',
                    help='Path to output previous training for' +
                    ' restoring weights')
parser.add_argument('-dp', '--data-path',
                    default='../data/processed/strain_point',
                    help='Path to dataset')

args = parser.parse_args()

savedir = os.path.join(args.output_path, 'models',
                       args.experiment_id)
loaddir = os.path.join(args.model_path, 'models', args.experiment_id)
if not os.path.exists(savedir):
    os.makedirs(savedir)
logdir = os.path.join(args.output_path, 'logs', args.experiment_id)
if not os.path.exists(logdir):
    os.makedirs(logdir)


def gridPlotTobuffer(warped_grid):
    fig, ax = plt.subplots()
    plotGrid(ax, warped_grid, color='purple')
    plt.gca().invert_yaxis()
    ax.set_title('Warp Grid')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return buf


# data_loader = DataLoader(args.data_path, shuffle=False)

defnet = DeformableNet(args.num_downsampling_layers, args.leakage)

writer = tf.contrib.summary.create_file_writer(logdir)
writer.set_as_default()

optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
global_step = tf.train.get_or_create_global_step()

try:
    defnet.load_weights(os.path.join(loaddir, args.experiment_id))
    print(f'Loaded weights from savedir')
    saved_gs = int(np.loadtxt(os.path.join(loaddir, 'global_step.txt')))
    global_step.assign(saved_gs)
    print(f'Starting training at step: {saved_gs}')
except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError):
    print('No previous weights found or weights couldn\'t be loaded')
    print('Starting fresh')

h5file = glob.glob(args.data_path + '/*/*.h5')[0]
with h5py.File(h5file) as data:
    fixed = tf.constant(data['tissue/data'][:][:15, :, :, None] / 255.,
                        dtype='float32')
    moving = tf.constant(data['tissue/data'][:][1:16, :, :, None] / 255.,
                         dtype='float32')

for i in range(args.num_steps):
    # fixed, moving = data_loader.loadSample()
    with tf.GradientTape() as tape:
        warped, warped_grid = defnet(fixed, moving)
        cross_corr = nncc2d(fixed, warped)
        bending_pen = defnet.bending_penalty
        loss = cross_corr + args.penalty * bending_pen
    grads = tape.gradient(loss, defnet.trainable_variables)

    print(f'{{"metric": "Loss", "value": {loss}, "step": {global_step.numpy()}}}')
    print(f'{{"metric": "NCC", "value": {cross_corr}, "step": {global_step.numpy()}}}')
    print(f'{{"metric": "Bending_penalty", "value": {args.penalty * bending_pen}, "step": {global_step.numpy()}}}')
    with tf.contrib.summary.record_summaries_every_n_global_steps(1):
        tf.contrib.summary.scalar('loss', loss)
        tf.contrib.summary.scalar('NCC loss', cross_corr)
        tf.contrib.summary.scalar('Bending penalty',
                                  args.penalty * bending_pen)
        for var in defnet.trainable_variables:
            tf.contrib.summary.histogram(var.name, var)
        for grad, var in zip(grads, defnet.trainable_variables):
            tf.contrib.summary.histogram(var.name + '/gradient', grad)
    with tf.contrib.summary.record_summaries_every_n_global_steps(10):
        tf.contrib.summary.image('Fixed', fixed[7:8, :, :, :], max_images=1)
        tf.contrib.summary.image('Moving', moving[7:8, :, :, :], max_images=1)
        tf.contrib.summary.image('Warped', warped[7:8, :, :, :], max_images=1)
        buf = gridPlotTobuffer(warped_grid)
        grid_img = tf.expand_dims(
            tf.image.decode_png(buf.getvalue(), channels=4), 0)
        tf.contrib.summary.image('Warp Grid', grid_img)

    optimizer.apply_gradients(zip(grads, defnet.trainable_variables),
                              global_step=global_step)

    defnet.save_weights(os.path.join(savedir, args.experiment_id))
    np.savetxt(os.path.join(savedir, 'global_step.txt'),
               [global_step.numpy()])
