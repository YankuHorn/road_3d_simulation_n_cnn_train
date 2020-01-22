import keras.backend as backend
import keras.layers as layers
import keras.utils as keras_utils
from keras.layers import Input, Flatten, Dense
from keras.models import Model
WEIGHTS_PATH_NO_TOP = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.5/'
    'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None,
              trainable=True):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    # if backend.image_data_format() == 'channels_first':
    #     bn_axis = 1
    # else:
    bn_axis = 3
    x = layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name, trainable=trainable)(x)
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name, trainable=trainable)(x)
    x = layers.Activation('relu', name=name)(x)
    return x


def InceptionV3(include_top=True,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=3,
                train_backbone=True,
                num_init_filters=8,
                **kwargs):
    """Instantiates the Inception v3 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    # global backend, layers, models, keras_utils
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    # if input_tensor is None:
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        # if not backend.is_keras_tensor(input_tensor):
        #     img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        # else:
        img_input = input_tensor

    # if backend.image_data_format() == 'channels_first':
    # channel_axis = 1
    # else:
    channel_axis = 3
    x = conv2d_bn(img_input, num_init_filters, 3, 3, strides=(2, 2), padding='valid', trainable=train_backbone)
    x = conv2d_bn(x, num_init_filters, 3, 3, padding='valid', trainable=train_backbone)
    x = conv2d_bn(x, num_init_filters * 2, 3, 3)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, int(num_init_filters * 2.5), 1, 1, padding='valid', trainable=train_backbone)
    x = conv2d_bn(x, num_init_filters * 6, 3, 3, padding='valid', trainable=train_backbone)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, num_init_filters * 2, 1, 1)

    branch5x5 = conv2d_bn(x, int(num_init_filters * 1.5), 1, 1)
    branch5x5 = conv2d_bn(branch5x5, num_init_filters * 2, 5, 5)

    branch3x3dbl = conv2d_bn(x, num_init_filters * 2, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, num_init_filters * 3, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, num_init_filters * 3, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, num_init_filters, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, num_init_filters * 2, 1, 1)

    branch5x5 = conv2d_bn(x, int(num_init_filters * 1.5), 1, 1)
    branch5x5 = conv2d_bn(branch5x5, num_init_filters * 2, 5, 5)

    branch3x3dbl = conv2d_bn(x, num_init_filters * 2, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, num_init_filters * 3, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, num_init_filters * 3, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, num_init_filters * 2, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, num_init_filters * 2, 1, 1)

    branch5x5 = conv2d_bn(x, int(num_init_filters * 1.5), 1, 1)
    branch5x5 = conv2d_bn(branch5x5, num_init_filters * 2, 5, 5)

    branch3x3dbl = conv2d_bn(x, num_init_filters * 2, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, num_init_filters * 3, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, num_init_filters * 3, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, num_init_filters * 2, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, num_init_filters * 12, 3, 3, strides=(2, 2), padding='valid', trainable=train_backbone)

    branch3x3dbl = conv2d_bn(x, num_init_filters * 2, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, num_init_filters * 3, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, num_init_filters * 3, 3, 3, strides=(2, 2), padding='valid', trainable=train_backbone)

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, num_init_filters * 6, 1, 1)

    branch7x7 = conv2d_bn(x, num_init_filters * 4, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, num_init_filters * 4, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, num_init_filters * 6, 7, 1)

    branch7x7dbl = conv2d_bn(x, num_init_filters * 4, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, num_init_filters * 4, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, num_init_filters * 4, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, num_init_filters * 4, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, num_init_filters * 6, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, num_init_filters * 6, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, num_init_filters * 6, 1, 1)

        branch7x7 = conv2d_bn(x, num_init_filters * 5, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, num_init_filters * 5, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, num_init_filters * 6, 7, 1)

        branch7x7dbl = conv2d_bn(x, num_init_filters * 5, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, num_init_filters * 5, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, num_init_filters * 5, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, num_init_filters * 5, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, num_init_filters * 6, 1, 7)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, num_init_filters * 6, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, num_init_filters * 6, 1, 1)

    branch7x7 = conv2d_bn(x, num_init_filters * 6, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, num_init_filters * 6, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, num_init_filters * 6, 7, 1)

    branch7x7dbl = conv2d_bn(x, num_init_filters * 6, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, num_init_filters * 6, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, num_init_filters * 6, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, num_init_filters * 6, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, num_init_filters * 6, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, num_init_filters * 6, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, num_init_filters * 6, 1, 1, trainable=True)
    branch3x3 = conv2d_bn(branch3x3, num_init_filters * 10, 3, 3,
                          strides=(2, 2), padding='valid', trainable=True)

    branch7x7x3 = conv2d_bn(x, num_init_filters * 6, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, num_init_filters * 6, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, num_init_filters * 6, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, num_init_filters * 6, 3, 3, strides=(2, 2), padding='valid', trainable=True)

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis,
        name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, num_init_filters * 10, 1, 1, trainable=True)

        branch3x3 = conv2d_bn(x, num_init_filters * 12, 1, 1, trainable=True)
        branch3x3_1 = conv2d_bn(branch3x3, num_init_filters * 12, 1, 3, trainable=True)
        branch3x3_2 = conv2d_bn(branch3x3, num_init_filters * 12, 3, 1, trainable=True)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, num_init_filters * 14, 1, 1, trainable=True)
        branch3x3dbl = conv2d_bn(branch3x3dbl, num_init_filters * 12, 3, 3, trainable=True)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, num_init_filters * 12, 1, 3, trainable=True)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, num_init_filters * 12, 3, 1, trainable=True)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, num_init_filters * 6, 1, 1, trainable=True)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    # model = Model(inputs, x, name='inception_v3')


    return x

def My_inception(input_shape,
                n_labels_scene_class,
                kernel,
                pool_size,
                filters_init_num=8,
                data_format="channels_last",
                output_mode="softmax",
                train_backbone=True):
    inputs = Input(shape=input_shape)
    x = InceptionV3(include_top=False, input_tensor=inputs,
                    input_shape=input_shape,
                    pooling=None,
                    classes=3,
                    train_backbone=True,
                    num_init_flters=filters_init_num)
    # flatten = Flatten()(x)
    gap = layers.GlobalAveragePooling2D()(x)
    #fc0 = Dense(273, activation='relu', name='fc0_273')(gap)
    fc1 = Dense(81, activation='relu', name='fc1_81_0114')(gap)
    fc2 = Dense(27, activation='relu', name='fc2_27')(fc1)
    fc3 = Dense(9, activation='relu', name='fc3_9')(fc2)
    # fc4 = Dense(3, activation='relu')(fc3)
    scene_class = Dense(3, activation='softmax', name='scene_class')(fc3)
    # finding horizon (segmentation)
    horizon = Dense(1, activation='linear', name='horizon')(fc3)
    # host_yaw_at_100m = Dense(1, activation='linear', name='host_yaw_at_100m')(fc3)
    # print("horizon shape", horizon.shape, "host_yaw_at_100m", host_yaw_at_100m, "scene_class shape", scene_class.shape)
    print("horizon shape", horizon.shape, "scene_class shape", scene_class.shape)

    my_inception_v3 = Model(inputs=inputs, outputs=[horizon, scene_class],
                                name="horizon_exit_merge")
    # weights_path = keras_utils.get_file(
    #     'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #     WEIGHTS_PATH_NO_TOP,
    #     cache_subdir='models',
    #     file_hash='bcbd6486424b2319ff4ef7d526e38f63')
    # my_inception_v3.load_weights(weights_path, by_name=True)

    return my_inception_v3

