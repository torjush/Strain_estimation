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


def unmaskedNncc2d(fixed, warped):
    fixed_mean = tf.reduce_mean(fixed, axis=[1, 2, 3], keep_dims=True)
    warped_mean = tf.reduce_mean(warped, axis=[1, 2, 3], keep_dims=True)

    N = tf.reduce_sum(tf.ones_like(fixed), axis=[1, 2, 3], keep_dims=True)
    fixed_variance = tf.div(tf.reduce_sum(tf.square(fixed - fixed_mean),
                                          axis=[1, 2, 3], keep_dims=True), N)
    warped_variance = tf.div(tf.reduce_sum(tf.square(warped - warped_mean),
                                           axis=[1, 2, 3], keep_dims=True), N)

    numerator = (fixed - fixed_mean) * (warped - warped_mean)
    denominator = tf.sqrt(fixed_variance * warped_variance)

    pixel_ncc = tf.div(numerator, denominator)
    ncc = tf.reduce_mean(pixel_ncc, axis=[1, 2, 3])
    return -tf.reduce_mean(ncc)
