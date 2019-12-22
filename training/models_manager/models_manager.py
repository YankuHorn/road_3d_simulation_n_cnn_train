from keras import optimizers
from training.models_manager.models.my_horizon_scene_class import my_horizon_scene_class_network
from training.models_manager.my_losses import my_L2_loss, my_scene_class_loss
RUN_ON_GPU = True


if not RUN_ON_GPU:
    import tensorflow as tf
    from keras import backend as K
    num_cores = 4
    num_CPU = 1
    num_GPU = 0
    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
                            device_count={'CPU': num_CPU, 'GPU': num_GPU})

    session = tf.Session(config=config)
    K.set_session(session)
    # K.set_floatx('float16')
    # K.set_epsilon(1e-4)
else:
    import tensorflow as tf

    # config = tf.ConfigProto()
    config = tf.compat.v1.ConfigProto()

    config.gpu_options.per_process_gpu_memory_fraction = 0.7


def batch_generator(training_generator):
    index = 0
    while True:
        # choose batch_size random images / labels from the data
        yield training_generator[index]
        index += 1
        if index >= len(training_generator):
            training_generator.on_epoch_end()
            index = 0


class ModelManager:
    def __init__(self):
        self.model = None
        self.logger_tensorboard = list()

    def get_model_for_train(self, model_name, n_labels, input_shape, init_filters_num, train_backbone):

        kernel = 3

        if model_name == "horizon_exit_merge":
            data_format = "channels_last"
            model = my_horizon_scene_class_network(input_shape=input_shape, n_labels_scene_class=n_labels, kernel=kernel,
                                                  pool_size=(2, 2), filters_init_num=init_filters_num, data_format=data_format,
                                                   output_mode="softmax", train_backbone=train_backbone)
        else:
            print('get_model_for_train is not familiar with model named', model_name)
            return False

        return model, data_format

    @staticmethod
    def get_model_for_inference(model_name, n_labels, input_shape, init_filters_num):
        kernel = 3
        data_format = 'channels_last'
        if model_name == 'horizon_exit_merge':
            model = my_horizon_scene_class_network(input_shape=input_shape, n_labels_scene_class=n_labels,
                                                   kernel=kernel,
                                                   pool_size=(2, 2), filters_init_num=init_filters_num,
                                                   data_format=data_format,
                                                   output_mode="softmax", train_backbone=train_backbone)
        else:
            print('get_model_for_inference is not familiar with model named', model_name)
            return False
        return model

    @staticmethod
    def compile_model(model, model_name, train_params):
        if model_name == "horizon_exit_merge":
            adam = optimizers.Adam(lr=train_params['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.9)
            horion_weight = train_params['weights']['horizon']
            scene_class_weight = train_params['weights']['scene_class']
            model.compile(loss={'horizon': my_L2_loss,
                                'scene_class': 'categorical_crossentropy'},
                                 loss_weights={'horizon': horion_weight, 'scene_class': scene_class_weight},
                                 optimizer=adam)
        elif model_name == "resnet_horizon":
            adam = optimizers.Adam(lr=train_params['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.9)
            model.compile(loss={'horizon': my_L2_loss,
                                'scene_class': 'categorical_crossentropy'},
                          optimizer=adam)
            # ResNet18(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
        else:
            print('compile_model not familiar with model named', model_name)
            return False
    @staticmethod
    def train_model(model, model_name, training_generator, validation_generator, train_params,
                    verbose=1, callbacks=None):
        """
        train the model, finds all relevant parameters according to model's name
        """
        if model_name == "horizon_exit_merge":
            # print(" ************************ type is: ", K.dtype(model.get_layer('conv2d_28').kernel))
            scene_class_weights = train_params['weights']["scene_class_weights"]
            history = model.fit_generator(generator=training_generator, epochs=1,
                                          validation_data=validation_generator,
                                          class_weight={'scene_class': scene_class_weights},
                                          verbose=verbose, use_multiprocessing=False, callbacks=callbacks)
        else:
            print('train_model is not familiar with model named', model_name)
            return False

        return history

    @staticmethod
    def print_model_data(history, model_name, i_epoch):
        if model_name == 'horizon_exit_merge':
            print(" ************************************************************************ ")
            print("Total       train loss ", history.history["loss"][0], "val loss ", history.history["val_loss"][0], "i_epoch=", i_epoch)
            print("horizon     train loss ", history.history["horizon_loss"][0], "val loss ", history.history["val_horizon_loss"][0], "i_epoch=", i_epoch)
            print("scene class train loss ", history.history["scene_class_loss"][0], "val loss ", history.history["val_scene_class_loss"][0], "i_epoch=", i_epoch)
        else:
            print('print_model_data is not familiar with model named', model_name)


    @staticmethod
    def get_outputs_names_for_model(model_name):
        res = list()
        if model_name == 'horizon_exit_merge':
            res.append('horizon')
            res.append('scene_class')
        elif model_name == 'resnet_horizon':
            res.append('horizon')
        else:
            print('get_outputs_names_for_model is not familiar with model named', model_name)
        return res
