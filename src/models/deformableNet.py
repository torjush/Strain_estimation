import tensorflow as tf
import numpy as np


class DeformableNet(tf.keras.Model):
    def __init__(self, num_conv_layers, leakage=0.2):
        super(DeformableNet, self).__init__(name='deformable_net')
        self.num_stages = num_conv_layers
        self.alpha = leakage
        self.__buildCNN()
        self.__bSplineKernels()

    def call(self, fixed, moving):
        height = int(moving.shape[1])
        width = int(moving.shape[2])
        batch_size = int(moving.shape[0])

        displacements = self.__runCNN(fixed, moving)
        self.interpolated_displacements = self.bSplineInterpolation(
            displacements, height, width)

        # make a grid of original points
        xx, yy = tf.meshgrid(tf.range(0., width),
                             tf.range(0., height))
        grid = tf.concat([tf.reshape(xx, [1, -1]),
                          tf.reshape(yy, [1, -1])], axis=0)
        grid = tf.stack([grid] * batch_size)

        # Add the interpolated displacements to the grid
        flat_grid = tf.reshape(grid, [batch_size, 2, -1])
        flat_displacements = tf.reshape(
            tf.transpose(self.interpolated_displacements, [0, 3, 1, 2]),
            [batch_size, 2, -1])

        warped_grid = tf.add(flat_displacements, flat_grid)

        warped = self.sampleBilinear(moving, warped_grid,
                                     height, width, batch_size)

        return warped, tf.reshape(warped_grid, [-1, 2, height, width])

    def trackPoints(self, fixed, moving, points):
        num_frames = fixed.shape[0] + 1
        tracked_points = np.zeros((num_frames, points.shape[0], 2))

        warped, warped_grid = self.call(fixed, moving)
        for j in range(points.shape[0]):
            x_coord = np.round(points[j, 0]).astype(int)
            y_coord = np.round(points[j, 1]).astype(int)

            tracked_points[0, j, 0] = x_coord
            tracked_points[0, j, 1] = y_coord
            for frame_num in range(num_frames - 1):
                # Find points in next frame
                next_x_coord = warped_grid[frame_num, 0, y_coord, x_coord]
                next_y_coord = warped_grid[frame_num, 1, y_coord, x_coord]
                next_x_coord = np.round(next_x_coord).astype(int)
                next_y_coord = np.round(next_y_coord).astype(int)

                # Update current points
                x_coord = next_x_coord
                y_coord = next_y_coord
                tracked_points[frame_num + 1, j, 0] = x_coord
                tracked_points[frame_num + 1, j, 1] = y_coord

        return tracked_points

    def __makeMeshgrids(self, nx, ny, width, height):
        x_num_between = tf.floor_div(width, nx) - 1
        y_num_between = tf.floor_div(height, ny) - 1

        x_step = 1 / tf.floor_div(width, nx)
        y_step = 1 / tf.floor_div(height, ny)

        x_range = tf.range(0., nx + x_step * x_num_between, x_step)[:width]
        x_range = tf.clip_by_value(x_range, 0., nx - 1)

        y_range = tf.range(0., ny + y_step * y_num_between, y_step)[:height]
        y_range = tf.clip_by_value(y_range, 0., ny - 1)

        xx, yy = tf.meshgrid(x_range, y_range)
        return xx, yy

    def sampleBilinear(self, img, warped_grid, height, width,
                       batch_size, epsilon=1e-5):
        x_t = tf.reshape(tf.slice(warped_grid, [0, 0, 0], [-1, 1, -1]),
                         [batch_size, height * width])
        y_t = tf.reshape(tf.slice(warped_grid, [0, 1, 0], [-1, 1, -1]),
                         [batch_size, height * width])

        # Find corners around each sampling point
        x0 = tf.floor(x_t)
        y0 = tf.floor(y_t)
        x1 = x0 + 1
        y1 = y0 + 1
        # Make sure we're within the bounds of the image
        x0 = tf.clip_by_value(x0, 0, width - 1)
        y0 = tf.clip_by_value(y0, 0, height - 1)
        x1 = tf.clip_by_value(x1, 0, width - 1)
        y1 = tf.clip_by_value(y1, 0, height - 1)

        # Find values of each corner
        def __makeGatherIndices(x, y):
            index_stack = []
            for batch in range(batch_size):
                indices = tf.stack([batch * tf.ones_like(x[batch, :]),
                                    y[batch, :],
                                    x[batch, :],
                                    tf.zeros_like(x[batch, :])])
                index_stack.append(indices)
            index_stack = tf.concat(index_stack, axis=1)
            index_stack = tf.transpose(index_stack)

            return tf.cast(index_stack, 'int32')

        Q1 = tf.gather_nd(img, __makeGatherIndices(x0, y0))
        Q1 = tf.reshape(Q1, [batch_size, height * width])
        Q2 = tf.gather_nd(img, __makeGatherIndices(x0, y1))
        Q2 = tf.reshape(Q2, [batch_size, height * width])
        Q3 = tf.gather_nd(img, __makeGatherIndices(x1, y0))
        Q3 = tf.reshape(Q3, [batch_size, height * width])
        Q4 = tf.gather_nd(img, __makeGatherIndices(x1, y1))
        Q4 = tf.reshape(Q4, [batch_size, height * width])

        # Do the actual interpolation
        R1 = ((x1 - x_t) / (x1 - x0 + epsilon)) * Q1 + \
            ((x_t - x0) / (x1 - x0 + epsilon)) * Q3
        R2 = ((x1 - x_t) / (x1 - x0 + epsilon)) * Q2 + \
            ((x_t - x0) / (x1 - x0 + epsilon)) * Q4

        warped_pixels = ((y1 - y_t) / (y1 - y0 + epsilon)) * R1 + \
            ((y_t - y0) / (y1 - y0 + epsilon)) * R2

        warped = tf.reshape(warped_pixels, [batch_size, height, width, 1])

        return warped

    def __bSplineKernels(self):
        upsample = self.num_stages * 2
        step = 1 / upsample
        u = tf.linspace(0., (upsample - 1) * step, upsample)
        v = tf.linspace(0., (upsample - 1) * step, upsample)

        def expandKernels(kernel, num_channels):
            # Nasty way of getting the filter to the shapes that
            # tf.nn.conv2d_transpose needs
            full_kernel = (lambda kernel=kernel, num_channels=num_channels:
                           [[[[kernel[i, j].numpy() if k == l else 0.
                               for k in range(num_channels)]
                              for l in range(num_channels)]
                             for j in range(kernel.shape[1])]
                            for i in range(kernel.shape[0])])()
            full_kernel = tf.constant(full_kernel)

            return full_kernel

        coeff_matrix = tf.constant([[-1., 3., -3., 1.],
                                    [3., -6., 3., 0.],
                                    [-3, 0., 3., 0.],
                                    [1., 4., 1., 0.]]) / 6.

        u_vecs = self.__BVectors(u)
        v_vecs = self.__BVectors(v)

        B_u = tf.einsum('jk, kl->jl', u_vecs, coeff_matrix)
        B_v = tf.einsum('jk, kl->jl', v_vecs, coeff_matrix)

        u_kernel = tf.reshape(tf.transpose(B_u[::-1, :]), [-1])
        v_kernel = tf.reshape(tf.transpose(B_v[::-1, :]), [-1])

        kernel = tf.matmul(tf.expand_dims(v_kernel, -1),
                           tf.expand_dims(u_kernel, 0))
        self.b_spline_kernel = expandKernels(kernel, 2)

        # Differentials for bending penalty
        dxdx_B_u = tf.einsum('jk, kl->jl',
                             self.__doubleDiffBVectors(u), coeff_matrix)
        dxdx_u_kernel = tf.reshape(tf.transpose(dxdx_B_u[::-1, :]), [-1])
        dxdx_kernel = tf.matmul(tf.expand_dims(v_kernel, -1),
                                tf.expand_dims(dxdx_u_kernel, 0))
        self.dxdx_kernel = expandKernels(dxdx_kernel, 2)

        dydy_B_v = tf.einsum('jk, kl->jl',
                             self.__doubleDiffBVectors(v), coeff_matrix)
        dydy_v_kernel = tf.reshape(tf.transpose(dydy_B_v[::-1, :]), [-1])
        dydy_kernel = tf.matmul(tf.expand_dims(dydy_v_kernel, -1),
                                tf.expand_dims(u_kernel, 0))
        self.dydy_kernel = expandKernels(dydy_kernel, 2)

        dx_B_u = tf.einsum('jk, kl->jl',
                           self.__diffBVectors(u), coeff_matrix)
        dx_u_kernel = tf.reshape(tf.transpose(dx_B_u[::-1, :]), [-1])
        dy_B_v = tf.einsum('jk, kl->jl',
                           self.__diffBVectors(v), coeff_matrix)
        dy_v_kernel = tf.reshape(tf.transpose(dy_B_v[::-1, :]), [-1])
        dxdy_kernel = tf.matmul(tf.expand_dims(dy_v_kernel, -1),
                                tf.expand_dims(dx_u_kernel, 0))
        self.dxdy_kernel = expandKernels(dxdy_kernel, 2)

    def bSplineInterpolation(self, displacements, new_height, new_width):
        num_channels = displacements.shape[3]
        batch_size = displacements.shape[0]

        upsample = self.num_stages * 2

        # Deal with odd dimensions
        if new_height % 2 == 1:
            y_pad = 1
        else:
            y_pad = 0
        if new_width % 2 == 1:
            x_pad = 1
        else:
            x_pad = 0
        displacements = tf.pad(displacements,
                               [[0, 0], [0, y_pad], [0, x_pad], [0, 0]])

        # Interpolate vector field
        res = tf.nn.conv2d_transpose(displacements, self.b_spline_kernel,
                                     output_shape=[batch_size,
                                                   new_height,
                                                   new_width,
                                                   num_channels],
                                     strides=[1, upsample, upsample, 1],
                                     padding='SAME')

        # Differentials for bending penalty
        dx_dx = tf.nn.conv2d_transpose(displacements, self.dxdx_kernel,
                                       output_shape=[batch_size,
                                                     new_height,
                                                     new_width,
                                                     num_channels],
                                       strides=[1, upsample, upsample, 1],
                                       padding='SAME')

        dy_dy = tf.nn.conv2d_transpose(displacements, self.dydy_kernel,
                                       output_shape=[batch_size,
                                                     new_height,
                                                     new_width,
                                                     num_channels],
                                       strides=[1, upsample, upsample, 1],
                                       padding='SAME')

        dx_dy = tf.nn.conv2d_transpose(displacements, self.dxdy_kernel,
                                       output_shape=[batch_size,
                                                     new_height,
                                                     new_width,
                                                     num_channels],
                                       strides=[1, upsample, upsample, 1],
                                       padding='SAME')

        self.bending_penalty = self.__bendingPenalty(dx_dx, dy_dy, dx_dy)
        return res

    def __bendingPenalty(self, dx_dx, dy_dy, dx_dy):
        dx_dx_2 = tf.reduce_sum(tf.square(dx_dx), axis=-1)
        dy_dy_2 = tf.reduce_sum(tf.square(dy_dy), axis=-1)
        dx_dy_2 = tf.reduce_sum(tf.square(dx_dy), axis=-1)

        summed = dx_dx_2 + dy_dy_2 + 2 * dx_dy_2

        # Approximate integrals by summing
        per_img_pen = tf.reduce_sum(summed, axis=[1, 2])

        return tf.reduce_mean(per_img_pen)

    def __BVectors(self, u):
        u = tf.expand_dims(u, -1)
        B_vecs = tf.concat([tf.pow(u, 3),
                            tf.square(u),
                            u,
                            tf.ones_like(u)],
                           axis=-1)
        return B_vecs

    def __doubleDiffBVectors(self, u):
        u = tf.expand_dims(u, -1)
        B_vecs = tf.concat([-6 * u + 6,
                            18 * u - 12,
                            -18 * u + 6,
                            6 * u],
                           axis=-1)
        return B_vecs

    def __diffBVectors(self, u):
        u = tf.expand_dims(u, -1)
        B_vecs = tf.concat([-3 * tf.square(u) + 6 * u - 3,
                            9 * tf.square(u) - 12 * u,
                            -9 * tf.square(u) + 6 * u + 3,
                            3*tf.square(u)],
                           axis=-1)
        return B_vecs

    def __buildCNN(self):
        # Convolutions + downsampling
        for i in range(self.num_stages):
            setattr(self, f'conv_{i}',
                    tf.keras.layers.Conv2D(
                        filters=32, kernel_size=[3, 3],
                        padding='same',
                        activation=None, use_bias=False))
            setattr(self, f'batchnorm_{i}',
                    tf.keras.layers.BatchNormalization())
            setattr(self, f'activation_{i}',
                    tf.keras.layers.LeakyReLU(alpha=self.alpha))
            setattr(self, f'avgpool_{i}',
                    tf.keras.layers.AveragePooling2D(pool_size=[2, 2]))

        # Final convolutions
        self.finalconv_0 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=[3, 3],
            padding='same',
            activation=None, use_bias=False)
        self.finalbatchnorm_0 = tf.keras.layers.BatchNormalization()
        self.finalactivation_0 = tf.keras.layers.LeakyReLU(alpha=self.alpha)

        self.finalconv_1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=[3, 3],
            padding='same',
            activation=None, use_bias=False)
        self.finalbatchnorm_1 = tf.keras.layers.BatchNormalization()
        self.finalactivation_1 = tf.keras.layers.LeakyReLU(alpha=self.alpha)

        # 1x1 convolutions
        self.conv1x1_0 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=[1, 1],
            padding='same',
            activation=None, use_bias=False)
        self.batchnorm1x1_0 = tf.keras.layers.BatchNormalization()
        self.activation1x1_0 = tf.keras.layers.LeakyReLU(alpha=self.alpha)

        self.conv1x1_1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=[1, 1],
            padding='same',
            activation=None, use_bias=False)
        self.batchnorm1x1_1 = tf.keras.layers.BatchNormalization()
        self.activation1x1_1 = tf.keras.layers.LeakyReLU(alpha=self.alpha)

        self.cnn_out = tf.keras.layers.Conv2D(
            filters=2, kernel_size=[1, 1],
            padding='same',
            activation=None)

    def __runCNN(self, fixed, moving):
        concatenated = tf.concat([fixed, moving], axis=-1)

        prev = concatenated
        for i in range(self.num_stages):
            prev = getattr(self, f'conv_{i}')(prev)
            prev = getattr(self, f'batchnorm_{i}')(prev)
            prev = getattr(self, f'activation_{i}')(prev)
            prev = getattr(self, f'avgpool_{i}')(prev)

        prev = self.finalconv_0(prev)
        prev = self.finalbatchnorm_0(prev)
        prev = self.finalactivation_0(prev)

        prev = self.finalconv_1(prev)
        prev = self.finalbatchnorm_1(prev)
        prev = self.finalactivation_1(prev)

        prev = self.conv1x1_0(prev)
        prev = self.batchnorm1x1_0(prev)
        prev = self.activation1x1_0(prev)

        prev = self.conv1x1_1(prev)
        prev = self.batchnorm1x1_1(prev)
        prev = self.activation1x1_1(prev)

        out = self.cnn_out(prev)

        return out
