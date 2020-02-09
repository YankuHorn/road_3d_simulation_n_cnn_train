
from keras.models import Model
from keras.layers.core import Activation # , Reshape
from keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Reshape
from keras.layers.merge import Concatenate
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


def my_hybrid_network(input_shape=(288, 70, 5), filters_init_num=16,
                                   data_format="channels_last", train_backbone=True):

    inputs = Input(shape=input_shape)
    filters_num = filters_init_num
    kernel_h = 3
    kernel_w = 3
    kernel = 3

    # encoder
    conv_1 = Convolution2D(filters_num, (kernel_h, kernel_w), padding="same", data_format=data_format, name='conv_2d_1',
                            trainable=train_backbone)(inputs[0])
    conv_1 = BatchNormalization(trainable=train_backbone, name='batch_normalization_1')(conv_1)
    conv_1 = Activation("relu")(conv_1)

    # conv_1a = Convolution2D(filters_num, (kernel_h, kernel_w), padding="same", data_format=data_format, name='conv_2d_1a',
    #                         trainable=train_backbone)(conv_1)
    # conv_1a = BatchNormalization(trainable=train_backbone, name='batch_normalization_1a')(conv_1a)
    # conv_1a = Activation("relu")(conv_1a)

    #concat_1 = Concatenate()([inputs, pool_1])
    pool_1 = MaxPooling2D((2, 2))(conv_1)
    filters_num = filters_init_num * 2

    conv_2 = Convolution2D(filters_num, (kernel_h, kernel_w), padding="same", data_format=data_format, name='conv_2d_2',
             trainable=train_backbone)(pool_1)
    conv_2 = BatchNormalization(trainable=train_backbone, name='batch_normalization_2')(conv_2)
    conv_2 = Activation("relu")(conv_2)

    # conv_2a = Convolution2D(filters_num, (kernel_h, kernel_w), padding="same", data_format=data_format,name='conv_2d_2a',
    #                        trainable=train_backbone)(conv_2)
    # conv_2a = BatchNormalization(trainable=train_backbone, name='batch_normalization_2a')(conv_2a)
    # conv_2a = Activation("relu")(conv_2a)

    pool_2 = MaxPooling2D((2, 2))(conv_2)

    # concat_2 = Concatenate()([conv_1, conv_2])

    filters_num = filters_init_num * 4
    conv_3 = Convolution2D(filters_num, (kernel_h, kernel_w), padding="same", data_format=data_format,
                           trainable=train_backbone, name='conv_2d_3',)(pool_2)
    conv_3 = BatchNormalization(trainable=train_backbone, name='batch_normalization_3')(conv_3)
    conv_3 = Activation("relu")(conv_3)

    # conv_3a = Convolution2D(filters_num, (kernel_h, kernel_w), padding="same", data_format=data_format,
    #                        trainable=train_backbone, name='conv_2d_3a')(conv_3)
    # conv_3a = BatchNormalization(trainable=train_backbone, name='batch_normalization_3a')(conv_3a)
    # conv_3a = Activation("relu")(conv_3a)

    pool_3 = MaxPooling2D((2, 2))(conv_3)
    pool_3a = AveragePooling2D((2, 2))(conv_3)
    concat_3 = Concatenate()([pool_3a, pool_3])
    # flatten_0 = Flatten()(pool_3)
    # fc_min1 = Dense(6912, activation='relu', name='fcmin_819')(flatten_0)
    # print(pool_3.shape)
    # back2bsnes = Reshape((36, 6, 32))(fc_min1)

    filters_num = filters_init_num * 8
    conv_4 = Convolution2D(filters_num, (kernel_h, kernel_w), padding="same", data_format=data_format,
                           trainable=train_backbone, name='conv_2d_4')(concat_3)
    conv_4 = BatchNormalization(trainable=train_backbone, name='batch_normalization_4')(conv_4)
    conv_4 = Activation("relu")(conv_4)


    # conv_4a = Convolution2D(filters_num, (kernel_h, kernel_w), padding="same", data_format=data_format,
    #                        trainable=train_backbone, name='conv_2d_4a')(conv_4)
    # conv_4a = BatchNormalization(trainable=train_backbone, name='batch_normalization_4a')(conv_4a)
    # conv_4a = Activation("relu")(conv_4a)
    pool_4 = MaxPooling2D((2, 2))(conv_4)
    pool_4a = AveragePooling2D((2, 2))(conv_4)
    concat_4 = Concatenate()([pool_4a, pool_4])
    filters_num = filters_init_num * 16

    conv_5 = Convolution2D(filters_num, (kernel_h, kernel_w), padding="same", data_format=data_format,
                           trainable=train_backbone, name='conv_2d_5')(concat_4)
    conv_5 = BatchNormalization(trainable=train_backbone, name='batch_normalization_5')(conv_5)
    conv_5 = Activation("relu")(conv_5)
    pool_5 = MaxPooling2D((3, 2))(conv_5)
    pool_5a = AveragePooling2D((3, 2))(conv_5)
    concat_5 = Concatenate()([pool_5a, pool_5])


    # # Classification
    flatten = Flatten()(concat_5)
    # fcm1 = Dense(819, activation='relu', name='fcn1_819')(flatten)
    fc0 = Dense(273, activation='relu', name='fc0_273_70')(flatten)
    fc1 = Dense(81, activation='relu', name='fc1_81')(fc0)
    fc2 = Dense(27, activation='relu', name='fc2_27')(fc1)
    fc3 = Dense(9, activation='relu', name='fc3_9')(fc2)

    # fc0 = Dense(273, activation='relu')(flatten)
    # fc1 = Dense(81, activation='relu')(fc0)
    # fc2 = Dense(27, activation='relu')(fc1)
    # fc3 = Dense(9, activation='relu')(fc2)
    # fc4 = Dense(3, activation='relu')(fc3)
    scene_class = Dense(3, activation='softmax', name='scene_class')(fc3)
    # finding horizon (segmentation)
    horizon = Dense(1, activation='linear', name='horizon')(fc3)
    host_yaw_at_100m = Dense(1, activation='linear', name='host_yaw_at_100m')(fc3)
    print("horizon shape", horizon.shape, "host_yaw_at_100m", host_yaw_at_100m, "scene_class shape", scene_class.shape)

    horizon_scene_class = Model(inputs=inputs, outputs=[horizon, host_yaw_at_100m, scene_class], name="horizon_exit_merge")

    return horizon_scene_class
