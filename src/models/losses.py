import tensorflow as tf


def nncc2d(fixed, warped):
    # Create a mask to pick out relevant pixels
    # The ultrasound images has a conical sector of
    # nonzero values, the rest should not be cared about
    mask = tf.not_equal(fixed, 0)
    mask_f = tf.cast(mask, 'float32')
    # Find number of pixels in sector
    N = tf.reduce_sum(mask_f,
                      axis=[1, 2, 3], keep_dims=True)
    epsilon = 1e-8

    masked_fixed_mean = tf.div(
        tf.reduce_sum(fixed, axis=[1, 2, 3], keep_dims=True), N)
    masked_warped_mean = tf.div(
        tf.reduce_sum(warped, axis=[1, 2, 3], keep_dims=True), N)

    warped_variance = tf.div(tf.reduce_sum(
        tf.square((warped - masked_warped_mean) * mask_f),
        axis=[1, 2, 3],
        keep_dims=True), N)
    fixed_variance = tf.div(tf.reduce_sum(
        tf.square((fixed - masked_fixed_mean) * mask_f),
        axis=[1, 2, 3],
        keep_dims=True), N)

    denominator = tf.sqrt(fixed_variance * warped_variance)
    numerator = tf.multiply((fixed - masked_fixed_mean) * mask_f,
                            (warped - masked_warped_mean) * mask_f)

    cc_imgs = tf.div(numerator, denominator + epsilon)
    cc = tf.div(tf.reduce_sum(cc_imgs, axis=[1, 2, 3], keep_dims=True), N)

    return -tf.reduce_mean(cc)


def bendingPenalty(displacements, mask=None):
    dx = tf.pad(displacements[:, :, 1:, :] - displacements[:, :, :-1, :],
                [[0, 0], [0, 0], [0, 1], [0, 0]], mode='CONSTANT')
    dy = tf.pad(displacements[:, 1:, :, :] - displacements[:, :-1, :, :],
                [[0, 0], [0, 1], [0, 0], [0, 0]])

    dx_dx = tf.pad(dx[:, :, 1:, :] - dx[:, :, :-1, :],
                   [[0, 0], [0, 0], [0, 1], [0, 0]], mode='CONSTANT')
    dy_dy = tf.pad(dy[:, 1:, :, :] - dy[:, :-1, :, :],
                   [[0, 0], [0, 1], [0, 0], [0, 0]], mode='CONSTANT')

    dx_dy = tf.pad(dy[:, :, 1:, :] - dy[:, :, :-1, :],
                   [[0, 0], [0, 0], [0, 1], [0, 0]], mode='CONSTANT')

    dx_dx_2 = tf.reduce_sum(tf.square(dx_dx), axis=-1)
    dy_dy_2 = tf.reduce_sum(tf.square(dy_dy), axis=-1)
    dx_dy_2 = tf.reduce_sum(tf.square(dx_dy), axis=-1)

    summed = dx_dx_2 + dy_dy_2 + 2 * dx_dy_2

    per_img_pen = tf.reduce_sum(summed, axis=[1, 2])

    return tf.reduce_mean(per_img_pen)
