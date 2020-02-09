import datetime
import yaml
# from src.utils import *
import _thread
import json
#import training.utils as utils
import os

from training.data.data_utils import DataUtils
from training.models_manager.models_manager import ModelManager
# from training.models_manager.my_losses import borders_loss_L2, borders_loss_L4
from training.train.logger import TrainLogger
from keras.callbacks import TensorBoard
from training.data.conv_data_generator import conv_DataGenerator
from training.data.pointnet_data_generator import PointNetDataGenerator
from training.data.hybrid_data_generator import DataGenerator
from training.data.new_hybrid_data_generator import My_new_hybrid_data_generator
from keras.models import load_model
import socket
from pathlib import Path
import numpy as np

# ML_LT_01:
# C:\Users\User\.conda\envs\tensorflow_gpu3\python.exe -m tensorboard.main --logdir=D:\phantomAI\data\synthesized_data\testing\2019_12_11__20_32_run\front_view_image\train_2019_07_09__13_32 --port=6041 --host=127.0.0.1


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent


def input_thread(a_list):
    input()
    a_list.append(True)


def save_model_thin(save_path, model, filename, val_loss, val_acc):

    # ------------------ Saving Model --------------------------

    # filename = dataset_name + date_time_str + "_nfltr" + str(filters_num) + "run_v_" + str(counter)

    model.save(os.path.join(save_path, filename + '.h5'))

    filehandler = open(os.path.join(save_path, filename + "history"), 'wb')

    filehandler.close()

    np.savez(os.path.join(save_path, filename + "_vectors"), val_loss=val_loss, val_acc=val_acc)


class Train:
    def __init__(self):

        self.data_manager = DataUtils()

        self.model_manager = ModelManager()
        projet_root = get_project_root()
        setup = os.path.join(projet_root, "training", "cfgs", "setup.json")
        with open(setup) as f:
            self.setup = json.load(f)

        param_file = os.path.join(projet_root, "training", "cfgs", "parameters.yml")
        with open(param_file) as f:
            self.params = yaml.load(f)

    def train(self):
        model_name = self.setup['model_name']
        main_img_dir = os.path.join(self.setup["base_input_data_dir"])

        date_time_str = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M")
        save_path = os.path.join(self.setup["results_dir"], model_name, "train_" + date_time_str)

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        print("Saving into: ", save_path)
        with open(os.path.join(save_path, "setup.json"), 'w') as fp:
            json.dump(self.setup, fp)
        with open(os.path.join(save_path, "parameters.yml"), 'w') as outfile:
            yaml.dump(self.params, outfile)


        # prepare model:
        n_labels = self.params['image_params']['n_labels']
        init_filters_num = self.params['train_params']['init_filters_num']

        img_height = self.params['image_params']['img_height']
        img_width = self.params['image_params']['img_width']
        n_channels = self.params['image_params']['channels']


        train_backbone = self.setup['train_backbone']
        # Training the model

        batch_size = self.params["train_params"]["batch_size"]  # 10

        max_iter = self.params["train_params"]["max_iter"]  # 160
        # augment = self.setup["augmentation"]
        # augment_params = self.setup["augmentation_params"]

        img_width = self.params["image_params"]["img_width"]
        img_height = self.params["image_params"]["img_height"]
        max_num_points = self.params["image_params"]["max_num_points"]
        max_columns = self.params["image_params"]["max_columns"]
        des_dim = (img_height, img_width)

        input_img_width = self.params["image_params"]["orig_img_width"]
        input_img_height = self.params["image_params"]["orig_img_height"]
        inp_dim = (input_img_height, input_img_width)

        raw_directory = os.path.join(main_img_dir, "train", "front_view_image")
        data_directory = os.path.join(main_img_dir, "train",  "meta_data")
        train_filesname = DataUtils.load_all_file_names(raw_directory, required_name_start='front_view_image', img_format="png")
        if len(train_filesname) == 0:
            print("No data in specified path: ", raw_directory, "Go to setup file and make sure the data directory, well, has data in it!")
        required_outputs = self.model_manager.get_outputs_names_for_model(model_name)

        data_format = "channels_last"
        # __init__(self, list_IDs, data_dir, labels_dir, batch_size=10, inp_dim=(560, 560), des_dim=(384, 384),
        #          n_channels=3,
        #          shuffle=True, augment=None, data_format="channels_last", outputs=None):
        if (model_name == "horizon_exit_merge") or (model_name == "inception_v3"):
            input_shape = (img_height, img_width, n_channels)
            training_generator = conv_DataGenerator(train_filesname, raw_directory, data_directory,
                                                    batch_size=batch_size, n_channels=n_channels,
                                                    shuffle=True, data_format=data_format, outputs=required_outputs)
        elif model_name == "pointnet_cls":
            input_shape = (max_num_points, n_channels)
            training_generator = PointNetDataGenerator(train_filesname, raw_directory, data_directory,
                                                       batch_size=batch_size, max_num_points=max_num_points,
                                                       shuffle=True, data_format=data_format, outputs=required_outputs)
        elif model_name == "hybrid":
            input_shape = (288, max_num_points, 2)
            training_generator = DataGenerator(train_filesname, raw_directory, data_directory,
                                                       batch_size=batch_size, max_num_columns=max_columns,
                                                       shuffle=True, data_format=data_format, outputs=required_outputs)
        elif model_name == "my_new_hybrid_network":
            input_shape = (288, max_num_points, 2)
            training_generator = My_new_hybrid_data_generator(train_filesname, raw_directory, data_directory,
                                               batch_size=batch_size, max_num_columns=max_columns,
                                               shuffle=True, data_format=data_format, outputs=required_outputs)

        val_data_directory = os.path.join(main_img_dir, "val", "meta_data")
        val_raw_directory = os.path.join(main_img_dir, "val", "front_view_image")
        val_filesname = DataUtils.load_all_file_names(val_raw_directory, required_name_start='front_view_image', img_format="png")
        if (model_name == "horizon_exit_merge") or (model_name == "inception_v3"):
            validation_generator = conv_DataGenerator(val_filesname, val_raw_directory, val_data_directory,
                                                      batch_size=batch_size, n_channels=n_channels,
                                                      shuffle=True, augment=None,
                                                      data_format=data_format, outputs=required_outputs)
        elif model_name == "pointnet_cls":
            validation_generator = PointNetDataGenerator(val_filesname, val_raw_directory, val_data_directory,
                                                         batch_size=batch_size, max_num_points=max_num_points,
                                                         shuffle=True, augment=None,
                                                         data_format=data_format, outputs=required_outputs)
        elif model_name == "hybrid":
            validation_generator = DataGenerator(val_filesname, val_raw_directory, val_data_directory,
                                                         batch_size=batch_size, max_num_columns=max_columns,
                                                         shuffle=True, augment=None,
                                                         data_format=data_format, outputs=required_outputs)
        elif model_name == "my_new_hybrid_network":
            validation_generator = My_new_hybrid_data_generator(val_filesname, val_raw_directory, val_data_directory,
                                                          batch_size=batch_size, max_num_columns=max_columns,
                                                          shuffle=True, data_format=data_format,
                                                          outputs=required_outputs)
        # temp_ids = train_filesname[:5]
        # training_generator.generate_data_with_list_ids(temp_ids)

        tensorboard = TensorBoard(log_dir=save_path,
                                  write_graph=True, write_images=False)

        logger_tensorboard1 = TrainLogger(os.path.join(save_path, 'train_loss'))
        logger_tensorboard2 = TrainLogger(os.path.join(save_path, 'val_loss'))

        logger_tensorboard3 = TrainLogger(os.path.join(save_path, 'train_horizon'))
        logger_tensorboard4 = TrainLogger(os.path.join(save_path, 'val_horizon'))
        #
        # logger_tensorboard5 = TrainLogger(os.path.join(save_path, 'train_yaw'))
        # logger_tensorboard6 = TrainLogger(os.path.join(save_path, 'val_yaw'))

        logger_tensorboard7 = TrainLogger(os.path.join(save_path, 'train_scene_class'))
        logger_tensorboard8 = TrainLogger(os.path.join(save_path, 'val_scene_class'))

        model, data_format = self.model_manager.get_model_for_train(model_name, n_labels, input_shape, init_filters_num, train_backbone)
        print(model.summary())
        if self.setup["load_model"]:
            presaved_model = self.setup["model_to_load"]
            model.load_weights(presaved_model, by_name=True)
        self.model_manager.compile_model(model, model_name, self.params['train_params'])

        i_epoch = 1
        dataset_name = str(main_img_dir.split('\\')[-1])
        filename_init = dataset_name + date_time_str + "_nfltr" + str(init_filters_num) + "run_v_"
        out_model_file = os.path.join(save_path, filename_init + "_model_file.py")
        import shutil
        shutil.copy('D:\\phantomAI\\code\\road_3d\\training\\models_manager\\models\\my_pointnet_horizon_HPInc.py', out_model_file)

        while i_epoch < max_iter:

            # history = self.model_manager.train_model(model, model_name, training_generator, validation_generator,
            #                                      self.params['train_params'], filename_init, save_path, callbacks=[tensorboard])
            # train_model(model, model_name, training_generator, validation_generator, train_params,
            #             verbose=1, callbacks=None)
            history = self.model_manager.train_model(model, model_name, training_generator, validation_generator,
                                                     self.params['train_params'], callbacks=[tensorboard])

            # Calculate metric
            # train_score = utils.evaluate_metrics(training_generator, model, input_shape, num_batches=2)
            # val_score = utils.evaluate_metrics(validation_generator, model, input_shape, num_batches=2)

            #print info:
            self.model_manager.print_model_data(history, model_name, i_epoch)

            # save to tensorboard:
            logger_tensorboard1.log_scalar('loss', history.history["loss"][0], i_epoch)
            logger_tensorboard2.log_scalar('loss', history.history["val_loss"][0], i_epoch)
            logger_tensorboard3.log_scalar('horizon', history.history["horizon_loss"][0], i_epoch)
            logger_tensorboard4.log_scalar('horizon', history.history["val_horizon_loss"][0], i_epoch)

            # logger_tensorboard5.log_scalar('yaw', history.history["yaw_loss"][0], i_epoch)
            # logger_tensorboard6.log_scalar('yaw', history.history["val_yaw_loss"][0], i_epoch)
            # logger_tensorboard7.log_scalar('scene_class', history.history["scene_class_loss"][0], i_epoch)
            # logger_tensorboard8.log_scalar('scene_class', history.history["val_scene_class_loss"][0], i_epoch)

            # save_path, seg_model, filename, val_loss, val_acc
            filename = filename_init + str(i_epoch)
            print("Saving into: ", save_path)
            save_model_thin(save_path, model, filename, val_loss=history.history["loss"][0], val_acc=0)

            log_fn = str(filename_init) + "_log.txt"
            log_full_fn = os.path.join(save_path, log_fn)
            with open(log_full_fn, 'a') as the_file:
                the_file.write(str(history.history["horizon_loss"][0]) + " " + str(history.history["val_horizon_loss"][0]) + '\n')

            print(" ************** @@@@@@ just fitted generator, i_epoch=", i_epoch)
            print("Saving into: ", save_path)
            i_epoch += 1


if __name__ == "__main__":
    print("333")
    tmh = Train()
    tmh.train()
