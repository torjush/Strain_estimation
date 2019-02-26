# TODO: Implement bending penalty
import tensorflow as tf


def nncc2d(moving_img, warped_img):
    moving_img = moving_img - tf.reduce_mean(moving_img,
                                             axis=[1, 2, 3],
                                             keep_dims=True)
    warped_img = warped_img - tf.reduce_mean(warped_img,
                                             axis=[1, 2, 3],
                                             keep_dims=True)

    epsilon = 1e-8
    warped_variance = tf.reduce_sum(tf.square(warped_img),
                                    axis=[1, 2, 3],
                                    keep_dims=True) + epsilon
    moving_variance = tf.reduce_sum(tf.square(moving_img),
                                    axis=[1, 2, 3],
                                    keep_dims=True) + epsilon

    denominator = tf.sqrt(moving_variance * warped_variance)
    numerator = tf.reduce_sum(tf.multiply(moving_img, warped_img),
                              axis=[1, 2, 3], keep_dims=True)

    cc = tf.div(numerator, denominator)

    return -tf.reduce_mean(cc)
