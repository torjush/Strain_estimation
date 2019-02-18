import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from deformableNet import DeformableNet
import tqdm

tf.enable_eager_execution()

logdir = '../../output/logs'

img_1 = np.zeros((250, 250), 'float32')
img_1[100:130, 55:90] = 255

img_2 = np.zeros((250, 250), 'float32')
img_2[110:120, 50:85] = 255

fixed_batch = np.concatenate([img_1[None, :, :, None]] * 4, axis=0)
moving_batch = np.concatenate([img_2[None, :, :, None]] * 4, axis=0)

defNet = DeformableNet(3)

writer = tf.contrib.summary.create_file_writer(logdir)
writer.set_as_default()

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
global_step = tf.train.get_or_create_global_step()
for i in tqdm.tqdm(range(5000)):
    with tf.GradientTape() as tape:
        warped = defNet(fixed_batch, moving_batch)
        loss = defNet.computeLoss(tf.constant(moving_batch), warped)
    grads = tape.gradient(loss, tf.trainable_variables())
    with tf.contrib.summary.record_summaries_every_n_global_steps(1):
        tf.contrib.summary.scalar('loss', loss)
        for var in tf.trainable_variables():
            tf.contrib.summary.histogram(var.name, var)
        for grad, var in zip(grads, tf.trainable_variables()):
            tf.contrib.summary.histogram(var.name + '/gradient', grad)

    optimizer.apply_gradients(zip(grads, tf.trainable_variables()),
                              global_step=global_step)
    global_step.assign_add(1)

    fig, ax = plt.subplots(ncols=3)
    ax[0].imshow(fixed_batch[0, :, :, 0])
    ax[1].imshow(moving_batch[0, :, :, 0])
    ax[2].imshow(warped[0, :, :, 0])
    plt.tight_layout()
    plt.savefig('imgs.png')
    plt.close()
