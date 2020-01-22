from keras import backend as K
from keras.layers import Input, Dropout, Flatten, Dense, MaxPooling2D, Dot, Lambda, \
    Reshape, BatchNormalization, Activation, Conv1D, AveragePooling2D
from keras.initializers import Constant
from keras.models import Model
from keras.regularizers import Regularizer
import keras.utils as keras_utils

import numpy as np


class OrthogonalRegularizer(Regularizer):
    """
    Considering that input is flattened square matrix X, regularizer tries to ensure that matrix X
    is orthogonal, i.e. ||X*X^T - I|| = 0. L1 and L2 penalties can be applied to it
    """
    def __init__(self, l1=0.0, l2=0.0):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)

    def __call__(self, x):
        size = int(np.sqrt(x.shape[1]))
        assert(size * size == x.shape[1])
        x = K.reshape(x, (-1, size, size))
        xxt = K.batch_dot(x, x, axes=(2, 2))
        regularization = 0.0
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(xxt - K.eye(size)))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(xxt - K.eye(size)))

        return regularization

    def get_config(self):
        return {'l1': float(self.l1), 'l2': float(self.l2)}


def orthogonal(l1=0.0, l2=0.0):
    """
    Functional wrapper for OrthogonalRegularizer.
    :param l1: l1 penalty
    :param l2: l2 penalty
    :return: Orthogonal regularizer to append to a loss function
    """
    return OrthogonalRegularizer(l1=l1, l2=l2)


def dense_bn(x, units, use_bias=True, scope=None, activation=None):
    """
    Utility function to apply Dense + Batch Normalization.
    """
    with K.name_scope(scope):
        x = Dense(units=units, use_bias=use_bias)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation(activation)(x)
    return x


def conv1d_bn(x, num_filters, kernel_size, padding='same', strides=1,
              use_bias=False, scope=None, activation='relu'):
    """
    Utility function to apply Convolution + Batch Normalization.
    """
    with K.name_scope(scope):
        input_shape = x.get_shape().as_list()[-2:]
        x = Conv1D(num_filters, kernel_size, strides=strides, padding=padding,
                   use_bias=use_bias, input_shape=input_shape)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation(activation)(x)
    return x


def transform_net(inputs, num_init_filters, scope=None, regularize=False):
    """
    Generates an orthogonal transformation tensor for the input data
    :param inputs: tensor with input image (either BxNxK or BxNx1xK)
    :param output_shape: shape of the ourput matrix
    :param scope: name of the grouping scope
    :param regularize: enforce orthogonality constraint
    :return: Bxoutput_shape[0]xoutput_shape[1] tensor of the transformation
    """
    with K.name_scope(scope):

        input_shape = inputs.get_shape().as_list()
        k = input_shape[-1]
        num_points = input_shape[-2]

        net = conv1d_bn(inputs, num_filters=num_init_filters, kernel_size=1, padding='valid',
                        use_bias=True, scope='tconv1')
        net = conv1d_bn(net, num_filters=num_init_filters * 2, kernel_size=1, padding='valid',
                        use_bias=True, scope='tconv2')
        net = conv1d_bn(net, num_filters=num_init_filters * 16, kernel_size=1, padding='valid',
                        use_bias=True, scope='tconv3')
        # net = conv1d_bn(net, num_filters=num_init_filters * 8, kernel_size=1, padding='valid',
        #                 use_bias=True, scope='tconv2')

        #  Done in 2D since 1D is painfully slow
        net = MaxPooling2D(pool_size=(num_points, 1), padding='valid')(Lambda(K.expand_dims)(net))
        net = Flatten()(net)

        net = dense_bn(net, units=num_init_filters * 8, scope='tfc1', activation='relu')
        net = dense_bn(net, units=num_init_filters * 4, scope='tfc2', activation='relu')

        # net = dense_bn(net, units=num_init_filters * 4, scope='tfc2', activation='relu')

        transform = Dense(units=k * k,
                          kernel_initializer='zeros', bias_initializer=Constant(np.eye(k).flatten()),
                          activity_regularizer=orthogonal(l2=0.001) if regularize else None)(net)
        transform = Reshape((k, k))(transform)

    return transform

#def sample_net(inputs, num_init_filters, scope='transform_net1', regularize=False):


def pointnet_base(inputs, num_init_filters=8, use_tnet=True):
    """
    Convolutional portion of pointnet, common across different tasks (classification, segmentation, etc)
    :param inputs: Input tensor with the point cloud shape (BxNxK)
    :param use_tnet: whether to use the transformation subnets or not.
    :return: tensor layer for CONV5 activations
    """

    # Obtain spatial point transform from inputs and convert inputs
    # sampled_points = sample_net(inputs, num_init_filters, scope='transform_net1', regularize=False)
    ptransform = transform_net(inputs, num_init_filters, scope='transform_net1', regularize=False)

    point_cloud_transformed = Dot(axes=(2, 1))([inputs, ptransform])

    # First block of convolutions
    net = conv1d_bn(point_cloud_transformed if use_tnet else inputs, num_filters=num_init_filters, kernel_size=1, padding='valid',
                    use_bias=True, scope='conv1')
    net = conv1d_bn(net, num_filters=num_init_filters, kernel_size=1, padding='valid',
                    use_bias=True, scope='conv2')

    # Obtain feature transform and apply it to the network
    if use_tnet:
        ftransform = transform_net(net, num_init_filters, scope='transform_net2', regularize=True)
        net_transformed = Dot(axes=(2, 1))([net, ftransform])

    # Second block of convolutions
    net = conv1d_bn(net_transformed if use_tnet else net, num_filters=num_init_filters, kernel_size=1, padding='valid',
                    use_bias=True, scope='conv3')
    net = conv1d_bn(net, num_filters=num_init_filters * 4, kernel_size=1, padding='valid',
                    use_bias=True, scope='conv4')
    net = conv1d_bn(net, num_filters=num_init_filters * 16, kernel_size=1, padding='valid',
                    use_bias=True, scope='conv5')
    # net = conv1d_bn(net, num_filters=num_init_filters * 8, kernel_size=1, padding='valid',
    #                 use_bias=True, scope='conv5')

    return net


def pointnet_cls(input_tensor=None, input_shape=(4096, 3), use_tnet=True, num_init_filters=16):

    assert K.image_data_format() == 'channels_last'

    # Generate input tensor and get base network
    if input_tensor is None:
        input_tensor = Input(input_shape, name='Input_cloud')
    num_point = input_tensor.shape[-2]
    net = pointnet_base(input_tensor, num_init_filters, use_tnet)
    net = MaxPooling2D(pool_size=(num_point, 1), padding='valid', name='maxpool')(Lambda(K.expand_dims)(net))
    net = Flatten()(net)
    fc0 = Dense(num_init_filters * 8, activation='relu')(net)
    fc1 = Dense(num_init_filters * 4, activation='relu')(fc0)
    fc2 = Dense(num_init_filters * 2, activation='relu')(fc1)
    fc3 = Dense(num_init_filters, activation='relu')(fc2)
    fc4 = Dense(num_init_filters // 2, activation='relu')(fc3)
    scene_class = Dense(3, activation='softmax', name='scene_class')(fc4)
    # finding horizon (segmentation)
    horizon = Dense(1, activation='linear', name='horizon')(fc4)
    print("horizon shape", horizon.shape)
    host_yaw_at_100m = Dense(1, activation='linear', name='host_yaw_at_100m')(fc4)
    model = Model(input_tensor, outputs=[horizon, scene_class, host_yaw_at_100m], name="my_pointnet_horizon_HPInc")

    return model