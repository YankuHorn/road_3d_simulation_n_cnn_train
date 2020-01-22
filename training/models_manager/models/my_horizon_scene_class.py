
from keras.models import Model
from keras.layers.core import Activation # , Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
# from keras.layers.merge import Concatenate
from keras.layers import Input, Flatten, Dense # , Lambda
# from keras.layers import LeakyReLU

#
# from keras import backend as K
#
# from keras.backend import argmax

# K.set_floatx('float16')
# K.set_epsilon(1e-4)

# seg_model = my_usegnet(input_shape=input_shape, n_labels=n_labels, kernel=kernel,
#                        pool_size=(2, 2), filters_init_num=4, output_mode="softmax")


def my_horizon_scene_class_network(
        input_shape,
        n_labels_scene_class,
        kernel,
        pool_size,
        filters_init_num=8,
        data_format="channels_last",
        output_mode="softmax",
        train_backbone=True):

    inputs = Input(shape=input_shape)
    filters_num = filters_init_num
    # encoder
    conv_1 = Convolution2D(filters_num, (kernel, kernel), dilation_rate=[3, 3], padding="same", data_format=data_format,
                            trainable=train_backbone, name='conv2d_1')(inputs)
                            #trainable = train_backbone, )(inputs)
    # conv_1 = BatchNormalization(trainable=train_backbone)(conv_1)
    conv_1 = BatchNormalization(trainable=train_backbone, name='batch_normalization_1')(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(filters_num, (kernel, kernel), dilation_rate=[3, 3], padding="same", data_format=data_format,
                           trainable=train_backbone, name='conv2d_2')(conv_1)
    conv_2 = BatchNormalization(trainable=train_backbone, name='batch_normalization_2')(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1 = MaxPooling2D((5, 5))(conv_2)
    filters_num = filters_init_num * 2

    conv_3 = Convolution2D(filters_num, (kernel, kernel), dilation_rate=[3, 3], padding="same", data_format=data_format,
                           trainable=train_backbone, name='conv2d_3')(pool_1)
    conv_3 = BatchNormalization(trainable=train_backbone, name='batch_normalization_3')(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(filters_num, (kernel, kernel), dilation_rate=[3, 3], padding="same", data_format=data_format,
                           trainable=train_backbone, name='conv2d_4')(conv_3)
    conv_4 = BatchNormalization(trainable=train_backbone, name='batch_normalization_4')(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2 = MaxPooling2D(5, 5)(conv_4)
    filters_num = filters_init_num * 4

    conv_5 = Convolution2D(filters_num, (kernel, kernel), dilation_rate=[2, 2], padding="same", data_format=data_format,
                           trainable=train_backbone, name='conv2d_5')(pool_2)
    conv_5 = BatchNormalization(trainable=train_backbone, name='batch_normalization_5')(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(filters_num, (kernel, kernel), dilation_rate=[2, 2], padding="same", data_format=data_format,
                           trainable=train_backbone, name='conv2d_6')(conv_5)
    conv_6 = BatchNormalization(trainable=train_backbone, name='batch_normalization_6')(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(filters_num, (kernel, kernel), dilation_rate=[2, 2], padding="same", data_format=data_format,
                           trainable=train_backbone, name='conv2d_7')(conv_6)
    conv_7 = BatchNormalization(trainable=train_backbone, name='batch_normalization_7')(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3 = MaxPooling2D(pool_size)(conv_7)
    filters_num = filters_init_num * 8

    conv_8 = Convolution2D(filters_num, (kernel, kernel), padding="same", data_format=data_format,
                           trainable=train_backbone, name='conv2d_8')(pool_3)
    conv_8 = BatchNormalization(trainable=train_backbone, name='batch_normalization_8')(conv_8)
    conv_8 = Activation("relu")(conv_8)
    # conv_9 = Convolution2D(filters_num, (kernel, kernel), padding="same", data_format=data_format,
    #                        trainable=train_backbone, name='conv2d_9')(conv_8)
    # conv_9 = BatchNormalization(trainable=train_backbone, name='batch_normalization_9')(conv_9)
    # conv_9 = Activation("relu")(conv_9)
    conv_10 = Convolution2D(filters_num, (kernel, kernel), padding="same", data_format=data_format,
                            trainable=train_backbone, name='conv2d_10')(conv_8)
    conv_10 = BatchNormalization(trainable=train_backbone, name='batch_normalization_10')(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4 = MaxPooling2D(pool_size)(conv_10)

    conv_11 = Convolution2D(filters_num, (kernel, kernel), padding="same", data_format=data_format,
                            trainable=train_backbone, name='conv2d_11')(pool_4)
    conv_11 = BatchNormalization(trainable=train_backbone, name='batch_normalization_11')(conv_11)
    conv_11 = Activation("relu")(conv_11)
    # conv_12 = Convolution2D(filters_num, (kernel, kernel), padding="same", data_format=data_format,
    #                         trainable=train_backbone, name='conv2d_12')(conv_11)
    # conv_12 = BatchNormalization(trainable=train_backbone, name='batch_normalization_12')(conv_12)
    # conv_12 = Activation("relu")(conv_12)
    conv_13 = Convolution2D(filters_num, (kernel, kernel), padding="same", data_format=data_format,
                            trainable=train_backbone, name='conv2d_13')(conv_11)
    conv_13 = BatchNormalization(trainable=train_backbone, name='batch_normalization_13')(conv_13)
    conv_13 = Activation("relu")(conv_13)

    # # Classification
    flatten = Flatten()(conv_13)
    fc0 = Dense(273, activation='relu', name='fc0_273')(flatten)
    fc1 = Dense(81, activation='relu', name='fc1_81_0114')(fc0)
    fc2 = Dense(27, activation='relu', name='fc2_27')(fc1)
    fc3 = Dense(9, activation='relu', name='fc3_9')(fc2)
    # fc4 = Dense(3, activation='relu')(fc3)
    scene_class = Dense(3, activation='softmax', name='scene_class')(fc3)
    # finding horizon (segmentation)
    horizon = Dense(1, activation='linear', name='horizon')(fc3)
    host_yaw_at_100m = Dense(1, activation='linear', name='host_yaw_at_100m')(fc3)
    print("horizon shape", horizon.shape, "host_yaw_at_100m", host_yaw_at_100m, "scene_class shape", scene_class.shape)

    horizon_scene_class = Model(inputs=inputs, outputs=[horizon, host_yaw_at_100m, scene_class], name="horizon_exit_merge")

    return horizon_scene_class
