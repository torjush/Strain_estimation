import tensorflow as tf
from models.deformableNet import DeformableNet
from models.losses import nncc2d
from data.dataLoader import DataLoader
import matplotlib.pyplot as plt
from misc.plots import plotGrid
import argparse
import os
import io
import tqdm

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
                       args.experiment_id, args.experiment_id)
logdir = os.path.join(args.output_path, 'logs', args.experiment_id)


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


data_loader = DataLoader(args.data_path, shuffle=True)

defnet = DeformableNet(args.num_downsampling_layers, args.leakage)

try:
    defnet.load_weights(savedir)
    print(f'Loaded weights from savedir')
except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError):
    print('No previous weights found or weights couldn\'t be loaded')
    print('Starting fresh')

writer = tf.contrib.summary.create_file_writer(logdir)
writer.set_as_default()

optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
global_step = tf.train.get_or_create_global_step()

for i in tqdm.trange(args.num_steps):
    fixed, moving = data_loader.loadSample()
    with tf.GradientTape() as tape:
        warped, warped_grid = defnet(fixed, moving)
        loss = nncc2d(moving, warped)
    grads = tape.gradient(loss, defnet.trainable_variables)

    with tf.contrib.summary.record_summaries_every_n_global_steps(1):
        tf.contrib.summary.scalar('loss', loss)
        for var in defnet.trainable_variables:
            tf.contrib.summary.histogram(var.name, var)
        for grad, var in zip(grads, defnet.trainable_variables):
            tf.contrib.summary.histogram(var.name + '/gradient', grad)
    with tf.contrib.summary.record_summaries_every_n_global_steps(10):
        tf.contrib.summary.image('Fixed', fixed, max_images=1)
        tf.contrib.summary.image('Moving', moving, max_images=1)
        tf.contrib.summary.image('Warped', warped, max_images=1)
        buf = gridPlotTobuffer(warped_grid)
        grid_img = tf.expand_dims(
            tf.image.decode_png(buf.getvalue(), channels=4), 0)
        tf.contrib.summary.image('Warp Grid', grid_img)

    optimizer.apply_gradients(zip(grads, defnet.trainable_variables),
                              global_step=global_step)

    if (i + 1) % 50 == 0:
        defnet.save_weights(savedir)
