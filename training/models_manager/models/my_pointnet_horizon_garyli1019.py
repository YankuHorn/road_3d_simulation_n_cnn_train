import tensorflow as tf
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Input, Dropout, Flatten, Dense, MaxPooling2D, Dot, Lambda, \
    Reshape, BatchNormalization, Activation, Conv1D, AveragePooling2D
from keras.initializers import Constant
from keras.models import Model
from keras.regularizers import Regularizer
import keras.utils as keras_utils

import numpy as np


def mat_mul(A, B):
    return tf.matmul(A, B)


def pointnet_cls(num_points=4096):

    input_points = Input(shape=(num_points, 3))
    x = Convolution1D(16, 1, activation='relu',
                      input_shape=(num_points, 3))(input_points)
    x = BatchNormalization()(x)
    x = Convolution1D(32, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(256, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=num_points)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(9, weights=[np.zeros([64, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
    input_T = Reshape((3, 3))(x)

    # forward net
    g = Lambda(mat_mul, arguments={'B': input_T})(input_points)
    g = Convolution1D(16, 1, input_shape=(num_points, 3), activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(16, 1, input_shape=(num_points, 3), activation='relu')(g)
    g = BatchNormalization()(g)

    # feature transform net
    f = Convolution1D(16, 1, activation='relu')(g)
    f = BatchNormalization()(f)
    f = Convolution1D(32, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Convolution1D(256, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    f = MaxPooling1D(pool_size=num_points)(f)
    f = Dense(128, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(64, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(16 * 16, weights=[np.zeros([64, 16 * 16]), np.eye(16).flatten().astype(np.float32)])(f)
    feature_T = Reshape((16, 16))(f)

    # forward net
    g = Lambda(mat_mul, arguments={'B': feature_T})(g)
    g = Convolution1D(16, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(32, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(256, 1, activation='relu')(g)
    g = BatchNormalization()(g)

    # global_feature
    global_feature = MaxPooling1D(pool_size=num_points)(g)

    # point_net_cls
    c = Dense(128, activation='relu')(global_feature)
    c = BatchNormalization()(c)
    c = Dropout(rate=0.7)(c)
    c = Dense(128, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(rate=0.7)(c)
    # --------------------------------------------------end of pointnet
    flatten = Flatten()(c)
    fc1 = Dense(81, activation='relu')(flatten)
    fc2 = Dense(27, activation='relu')(fc1)
    fc3 = Dense(9, activation='relu')(fc2)
    # fc4 = Dense(3, activation='relu')(fc3)
    scene_class = Dense(3, activation='softmax', name='scene_class')(fc3)
    # finding horizon (segmentation)
    horizon = Dense(1, activation='linear', name='horizon')(fc3)
    host_yaw_at_100m = Dense(1, activation='linear', name='host_yaw_at_100m')(fc3)
    print("horizon shape", horizon.shape, "host_yaw_at_100m", host_yaw_at_100m, "scene_class shape", scene_class.shape)

    horizon_scene_class = Model(inputs=input_points, outputs=[horizon, host_yaw_at_100m, scene_class],
                                name="horizon_exit_merge")
    # print(horizon_scene_class.summary())
    return horizon_scene_class

