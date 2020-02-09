print("MHSC 0")
from keras.models import Model
from keras.layers.core import Activation # , Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
# from keras.layers.merge import Concatenate
from keras.layers import Input, Flatten, Dense # , Lambda
# from keras.layers import LeakyReLU
print("MHSC 1")
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
    conv_1 = Convolution2D(filters_num, (kernel, kernel), padding="same", data_format=data_format,
                            trainable=train_backbone)(inputs)
    conv_1 = BatchNormalization(trainable=train_backbone)(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(filters_num, (kernel, kernel), padding="same", data_format=data_format, trainable=train_backbone)(conv_1)
    conv_2 = BatchNormalization(trainable=train_backbone)(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1 = MaxPooling2D((3, 3))(conv_2)
    filters_num = filters_init_num * 2

    conv_3 = Convolution2D(filters_num, (kernel, kernel), padding="same", data_format=data_format,  trainable=train_backbone)(pool_1)
    conv_3 = BatchNormalization(trainable=train_backbone)(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(filters_num, (kernel, kernel), padding="same", data_format=data_format,  trainable=train_backbone)(conv_3)
    conv_4 = BatchNormalization(trainable=train_backbone)(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2 = MaxPooling2D(3, 3)(conv_4)
    filters_num = filters_init_num * 4

    conv_5 = Convolution2D(filters_num, (kernel, kernel), padding="same", data_format=data_format,  trainable=train_backbone)(pool_2)
    conv_5 = BatchNormalization(trainable=train_backbone)(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(filters_num, (kernel, kernel), padding="same", data_format=data_format,  trainable=train_backbone)(conv_5)
    conv_6 = BatchNormalization(trainable=train_backbone)(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(filters_num, (kernel, kernel), padding="same", data_format=data_format,  trainable=train_backbone)(conv_6)
    conv_7 = BatchNormalization(trainable=train_backbone)(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3 = MaxPooling2D(pool_size)(conv_7)
    filters_num = filters_init_num * 8

    conv_8 = Convolution2D(filters_num, (kernel, kernel), padding="same", data_format=data_format,  trainable=train_backbone)(pool_3)
    conv_8 = BatchNormalization(trainable=train_backbone)(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Convolution2D(filters_num, (kernel, kernel), padding="same", data_format=data_format,  trainable=train_backbone)(conv_8)
    conv_9 = BatchNormalization(trainable=train_backbone)(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Convolution2D(filters_num, (kernel, kernel), padding="same", data_format=data_format,  trainable=train_backbone)(conv_9)
    conv_10 = BatchNormalization(trainable=train_backbone)(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4 = MaxPooling2D(pool_size)(conv_10)
    conv_14 = Convolution2D(filters_num, (kernel, kernel), padding="same", data_format=data_format, trainable=train_backbone)(pool_4)
    conv_14 = BatchNormalization(trainable=train_backbone)(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Convolution2D(filters_num, (kernel, kernel), padding="same", data_format=data_format, trainable=train_backbone)(conv_14)
    conv_15 = BatchNormalization(trainable=train_backbone)(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Convolution2D(filters_num, (kernel, kernel), padding="same", data_format=data_format, trainable=train_backbone)(conv_15)
    conv_16 = BatchNormalization(trainable=train_backbone)(conv_16)
    conv_16 = Activation("relu")(conv_16)

    # # Classification
    flatten = Flatten()(conv_16)
    fc1 = Dense(81, activation='relu')(flatten)
    fc2 = Dense(27, activation='relu')(fc1)
    fc3 = Dense(9, activation='relu')(fc2)
    # fc3_ = Dense(9, activation='relu')(fc3)
    # fc3__ = Dense(9, activation='relu')(fc3_)
    # fc4 = Dense(3, activation='relu')(fc3)
    scene_class = Dense(3, activation='softmax', name='scene_class')(fc3)
    # finding horizon (segmentation)
    horizon = Dense(1, activation='linear', name='horizon')(fc3)
    print("horizon shape", horizon.shape, "scene_class shape", scene_class.shape)

    horizon_scene_class = Model(inputs=inputs, outputs=[horizon, scene_class], name="horizon_exit_merge")

    return horizon_scene_class