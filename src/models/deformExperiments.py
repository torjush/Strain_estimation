import tensorflow as tf
import numpy as np
from deformableNet import DeformableNet
from losses import nncc2d
import tqdm
import argparse
import os

tf.enable_eager_execution()
parser = argparse.ArgumentParser()
parser.add_argument('-id', '--experiment-id',
                    default='deform')
parser.add_argument('-lr', '--learning-rate',
                    default=1e-3, type=float)
parser.add_argument('-ds', '--num-downsampling-layers',
                    default=1, type=int)
parser.add_argument('-p', '--model-path',
                    default='../../output/models/',
                    help='Model will be saved/restored from $model_path/$experiment_id')
parser.add_argument('-gs', '--global-step',
                    default=None, type=int)
args = parser.parse_args()

model_path = os.path.join(args.model_path,
                          args.experiment_id, args.experiment_id)
logdir = '../../output/logs/' + args.experiment_id

img_1 = np.random.randint(0, 100, (250, 500)).astype('float32')

img_2 = img_1.copy()
img_1[100:130, 55:90] = 255

img_2[105:125, 60:85] = 255

fixed_batch = tf.constant(np.concatenate([img_1[None, :, :, None]] * 4, axis=0))
moving_batch = tf.constant(np.concatenate([img_2[None, :, :, None]] * 4, axis=0))

defNet = DeformableNet(args.num_downsampling_layers)
try:
    defNet.load_weights(model_path)
    print(f"Loaded weights from {model_path}")
except:
    print('No previous weights found or weights couldn\'t be loaded')
    print('Starting fresh')

writer = tf.contrib.summary.create_file_writer(logdir)
writer.set_as_default()

optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
global_step = tf.train.get_or_create_global_step()
if args.global_step:
    global_step.assign(args.global_step)
for i in tqdm.tqdm(range(5000)):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(fixed_batch)
        tape.watch(moving_batch)
        warped = defNet(fixed_batch, moving_batch)
        loss = nncc2d(moving_batch, warped)
    grads = tape.gradient(loss, defNet.trainable_variables)
    del tape
    with tf.contrib.summary.record_summaries_every_n_global_steps(1):
        tf.contrib.summary.scalar('loss', loss)
        for var in defNet.trainable_variables:
            tf.contrib.summary.histogram(var.name, var)
        for grad, var in zip(grads, defNet.trainable_variables):
            tf.contrib.summary.histogram(var.name + '/gradient', grad)
    with tf.contrib.summary.record_summaries_every_n_global_steps(50):
        tf.contrib.summary.image('Fixed', fixed_batch, max_images=1)
        tf.contrib.summary.image('Moving', moving_batch, max_images=1)
        tf.contrib.summary.image('Warped', warped, max_images=1)

    optimizer.apply_gradients(zip(grads, defNet.trainable_variables),
                              global_step=global_step)
    if (i + 1) % 100 == 0:
        defNet.save_weights(model_path)
