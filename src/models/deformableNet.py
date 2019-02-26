import tensorflow as tf


class DeformableNet():
    # TODO: subclass tf.keras.Model
    def __init__(self, num_conv_layers, leakage=0.2):
        self.cnn = tf.make_template('convnet',
                                    self.__buildCNN,
                                    num_stages=num_conv_layers,
                                    alpha=leakage)

    def __call__(self, fixed, moving):
        height = moving.shape[1]
        width = moving.shape[2]
        batch_size = moving.shape[0]

        points = self.cnn(fixed, moving)
        interpolated_points = self.bSplineInterpolation(points, height, width)

        # make a grid of original points
        xx, yy = self.__makeMeshgrids(width, height, width, height)
        grid = tf.concat([tf.reshape(xx, [1, -1]),
                          tf.reshape(yy, [1, -1])], axis=0)
        grid = tf.stack([grid] * batch_size)

        # Add x and y components of the interpolated points to the grid
        flat_grid = tf.reshape(grid, [batch_size, 2, -1])
        flat_points = tf.reshape(
            tf.transpose(interpolated_points, [0, 3, 1, 2]),
            [batch_size, 2, -1])

        transformation_grid = tf.add(flat_points, flat_grid)

        warped = self.sampleBilinear(fixed, transformation_grid,
                                     height, width, batch_size)

        return warped

    def __makeMeshgrids(self, nx, ny, width, height):
        x_range = tf.linspace(0., nx - 1, width)
        y_range = tf.linspace(0., ny - 1, height)

        xx, yy = tf.meshgrid(x_range, y_range)

        return xx, yy

    def sampleBilinear(self, img, trans_grid, height, width,
                       batch_size, epsilon=1e-5):
        x_t = tf.reshape(tf.slice(trans_grid, [0, 0, 0], [-1, 1, -1]),
                         [batch_size, height * width])
        y_t = tf.reshape(tf.slice(trans_grid, [0, 1, 0], [-1, 1, -1]),
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

    def bSplineInterpolation(self, points, new_height, new_width):
        # TODO: Implement using transposed convolutions and fixed kernels
        nx = tf.cast(points.shape[2], 'float32')
        ny = tf.cast(points.shape[1], 'float32')
        num_channels = points.shape[3]
        batch_size = points.shape[0]

        xx, yy = self.__makeMeshgrids(nx, ny, new_height, new_width)

        u = tf.div(xx, nx) - tf.floor_div(xx, nx)
        v = tf.div(yy, ny) - tf.floor_div(yy, ny)

        i = tf.floor(xx)
        j = tf.floor(yy)

        padded_points = tf.pad(points, [[0, 0], [1, 3], [1, 3], [0, 0]],
                               'SYMMETRIC')

        def __makeGatherIndices(i, j):
            flat_i = tf.reshape(i, [-1])
            flat_j = tf.reshape(j, [-1])

            index_stack = []
            # TODO: Fiks denne med tf.range i stedet for loops
            for batch in range(batch_size):
                for channel in range(num_channels):
                    stacked = tf.stack([batch * tf.ones_like(flat_i),
                                        flat_j,
                                        flat_i,
                                        channel * tf.ones_like(flat_i)])
                    index_stack.append(stacked)

            index_stack = tf.concat(index_stack, axis=1)
            index_stack = tf.transpose(index_stack)
            index_stack = tf.cast(index_stack, 'int32')

            return index_stack

        points_matrix = []
        for add_j in range(4):
            points_row = []
            for add_i in range(4):
                gather_idc = self.__getGatherIndices(i + add_i, j + add_j,
                                                     batch_size, num_channels)
                points_entry = tf.gather_nd(padded_points, gather_idc)
                points_entry = tf.reshape(points_entry,
                                          [batch_size, new_height,
                                           new_width, num_channels])
                points_entry = tf.expand_dims(points_entry, -1)
                points_row.append(points_entry)
            points_row = tf.stack(points_row, -1)
            points_matrix.append(points_row)
        points_matrix = tf.concat(points_matrix, -2)
        points_matrix = tf.cast(points_matrix, 'float32')

        B_u = self.__constructBMatrix(u, batch_size, num_channels,
                                      new_height, new_width)
        B_v = self.__constructBMatrix(v, batch_size, num_channels,
                                      new_height, new_width)
        interm = tf.matmul(B_u, points_matrix)
        res = tf.matmul(interm, B_v, transpose_b=True)
        res = tf.reshape(res,
                         [batch_size, new_height, new_width, num_channels])

        return res

    def __constructBMatrix(self, u, batch_size, num_channels,
                           new_height, new_width):
        u = tf.expand_dims(u, 0)
        u = tf.expand_dims(u, -1)
        u = tf.tile(u, [batch_size, 1, 1, num_channels])
        expanded_u = tf.expand_dims(u, -1)
        # Construct vectors [u^3 u^2 u 1]^T
        u_vecs = tf.concat([tf.pow(expanded_u, 3),
                            tf.square(expanded_u),
                            expanded_u,
                            tf.ones_like(expanded_u)],
                           axis=-1)
        # Make them row vectors
        u_vecs = tf.expand_dims(u_vecs, -2)
        coeff_matrix = tf.tile(
            tf.constant([[[[[[-1., 3., -3., 1.],
                             [3., -6., 3., 0.],
                             [-3, 0., 3., 0.],
                             [1., 4., 1., 0.]]]]]]),
            [batch_size, new_height, new_width, num_channels, 1, 1])
        B_u = tf.matmul(u_vecs, coeff_matrix) * (1 / 6.)

        return B_u

    def __buildCNN(self, fixed, moving, num_stages, alpha):
        concatenated = tf.concat([fixed, moving], axis=-1)

        # Convolutions + downsampling
        prev = concatenated
        for i in range(num_stages):
            prev = tf.layers.Conv2D(
                filters=32, kernel_size=[3, 3],
                padding='same',
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer())(prev)
            prev = tf.nn.leaky_relu(prev, alpha=alpha)
            prev = tf.layers.AveragePooling2D(pool_size=[2, 2],
                                              strides=2)(prev)
            prev = tf.layers.BatchNormalization()(prev)

        # Final convolutions
        prev = tf.layers.Conv2D(
            filters=32, kernel_size=[3, 3],
            padding='same',
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer())(prev)
        prev = tf.nn.leaky_relu(prev, alpha=alpha)
        prev = tf.layers.BatchNormalization()(prev)
        prev = tf.layers.Conv2D(
            filters=32, kernel_size=[3, 3],
            padding='same',
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer())(prev)
        prev = tf.nn.leaky_relu(prev, alpha=alpha)
        prev = tf.layers.BatchNormalization()(prev)

        # 1x1 convolutions
        prev = tf.layers.Conv2D(
            filters=32, kernel_size=[1, 1],
            padding='same',
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer())(prev)
        prev = tf.nn.leaky_relu(prev, alpha=alpha)
        prev = tf.layers.BatchNormalization()(prev)
        prev = tf.layers.Conv2D(
            filters=32, kernel_size=[1, 1],
            padding='same',
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer())(prev)
        prev = tf.nn.leaky_relu(prev, alpha=alpha)
        prev = tf.layers.BatchNormalization()(prev)

        out = tf.layers.Conv2D(
            filters=2, kernel_size=[1, 1],
            padding='same',
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer())(prev)

        return out
