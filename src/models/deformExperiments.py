import tensorflow as tf
import numpy as np
from deformableNet import DeformableNet
from losses import nncc2d
import tqdm
import argparse

tf.enable_eager_execution()
parser = argparse.ArgumentParser()
parser.add_argument('-id', '--experiment-id',
                    default='deform')
parser.add_argument('-lr', '--learning-rate',
                    default=1e-3, type=float)
args = parser.parse_args()

logdir = '../../output/logs/' + args.experiment_id

img_1 = np.zeros((250, 250), 'float32')
img_1[100:130, 55:90] = 255

img_2 = np.zeros((250, 250), 'float32')
img_2[110:120, 50:85] = 255

fixed_batch = np.concatenate([img_1[None, :, :, None]] * 4, axis=0)
moving_batch = np.concatenate([img_2[None, :, :, None]] * 4, axis=0)

defNet = DeformableNet(3)

writer = tf.contrib.summary.create_file_writer(logdir)
writer.set_as_default()

optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
global_step = tf.train.get_or_create_global_step()
for i in tqdm.tqdm(range(5000)):
    with tf.GradientTape() as tape:
        warped = defNet(fixed_batch, moving_batch)
        loss = nncc2d(tf.constant(moving_batch), warped)
    grads = tape.gradient(loss, tf.trainable_variables())
    with tf.contrib.summary.record_summaries_every_n_global_steps(1):
        tf.contrib.summary.scalar('loss', loss)
        for var in tf.trainable_variables():
            tf.contrib.summary.histogram(var.name, var)
        for grad, var in zip(grads, tf.trainable_variables()):
            tf.contrib.summary.histogram(var.name + '/gradient', grad)
    with tf.contrib.summary.record_summaries_every_n_global_steps(50):
        tf.contrib.summary.image('Fixed', fixed_batch, max_images=1)
        tf.contrib.summary.image('Moving', moving_batch, max_images=1)
        tf.contrib.summary.image('Warped', warped, max_images=1)

    optimizer.apply_gradients(zip(grads, tf.trainable_variables()),
                              global_step=global_step)
    global_step.assign_add(1)
