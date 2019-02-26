import tensorflow as tf
from stn import spatial_transformer_network as transformer
import numpy as np


class AffineNet():
    # TODO: Subclass tf.keras.Model
    def __init__(self, num_conv_layers):
        # Make convnet as template to share variables
        # between moving and fixed image pipelines
        self.convnet = tf.make_template('convnet',
                                        self.__buildConvnet,
                                        num_stages=num_conv_layers)
        self.parameter_regressor = tf.make_template('param_regressor',
                                                    self.__parameterRegressor)

    def __call__(self, fixed, moving):
        fixed_features = self.convnet(fixed)
        moving_features = self.convnet(moving)

        params = self.parameter_regressor(fixed_features, moving_features)
        transformation_matrix = self.__transformationMatrix(params)

        warped = transformer(fixed, transformation_matrix)
        return warped

    def __buildConvnet(self, imgs, num_stages):
        initializer = tf.contrib.layers.xavier_initializer()
        prev = imgs
        for i in range(num_stages):
            prev = tf.layers.Conv2D(filters=32, kernel_size=[3, 3],
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer=initializer)(prev)
            # Global pooling in last stage
            if i < num_stages - 1:
                prev = tf.layers.AveragePooling2D(pool_size=[2, 2],
                                                  strides=1)(prev)

        global_pooled = tf.reduce_mean(prev, axis=[1, 2])
        return global_pooled

    def __parameterRegressor(self, fixed, moving):
        initializer = tf.contrib.layers.xavier_initializer()
        merged = tf.concat([fixed, moving], axis=-1)
        hidden = tf.layers.Dense(32, activation='relu',
                                 kernel_initializer=initializer)(merged)
        # Translation parameters
        t_regressor = tf.layers.Dense(
            2, kernel_initializer=initializer)(hidden)

        # Scaling parameters
        s_regressor = tf.layers.Dense(
            2, kernel_initializer=initializer)(hidden)
        s_regressor = tf.clip_by_value(s_regressor, 0.5, 1.5)

        # Shearing parameters
        sh_regressor = tf.layers.Dense(
            2, kernel_initializer=initializer)(hidden)
        sh_regressor = tf.clip_by_value(sh_regressor, -np.pi, np.pi)

        # Rotation parameter
        theta_regressor = tf.layers.Dense(
            2, kernel_initializer=initializer)(hidden)
        theta_regressor = tf.clip_by_value(theta_regressor, -np.pi, np.pi)

        params = tf.concat(
            [t_regressor, s_regressor, sh_regressor, theta_regressor],
            axis=-1)

        return params

    def __transformationMatrix(self, params):
        # Define each transformation matrix entry and pack them
        zerozero = (params[:, 2] * tf.cos(params[:, 6]) +
                    params[:, 5] * params[:, 3] * tf.sin(params[:, 6]))
        zeroone = (params[:, 4] * params[:, 3] * tf.cos(params[:, 6]) -
                   params[:, 2] * tf.sin(params[:, 6]))
        zerotwo = (params[:, 2] * params[:, 0] +
                   params[:, 4] * params[:, 3] * params[:, 1])
        onezero = (params[:, 5] * params[:, 2] * tf.cos(params[:, 6]) +
                   params[:, 3] * tf.sin(params[:, 6]))
        oneone = (params[:, 3] * tf.cos(params[:, 6]) -
                  params[:, 5] * params[:, 2] * tf.sin(params[:, 6]))
        onetwo = (params[:, 5] * params[:, 2] * params[:, 0] +
                  params[:, 3] * params[:, 1])

        transformation_matrix = tf.stack([zerozero, zeroone, zerotwo,
                                          onezero, oneone, onetwo], axis=-1)
        return transformation_matrix
