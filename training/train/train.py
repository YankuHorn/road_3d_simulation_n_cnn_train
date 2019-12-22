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
from training.data.data_generator import DataGenerator
from keras.models import load_model
import socket
from pathlib import Path
import numpy as np


# ML_LT_01:
# C:\Users\kobih\Anaconda3\envs\python3\python.exe -m tensorboard.main --logdir=D:\phantomAI\data\synthesized_data\testing\2019_12_11__20_32_run\front_view_image\train_2019_07_09__13_32 --port=6041 --host=127.0.0.1

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
        input_shape = (img_width, img_height, n_channels)

        train_backbone = self.setup['train_backbone']
        # Training the model

        batch_size = self.params["train_params"]["batch_size"]  # 10

        max_iter = self.params["train_params"]["max_iter"]  # 160
        # augment = self.setup["augmentation"]
        # augment_params = self.setup["augmentation_params"]

        img_width = self.params["image_params"]["img_width"]
        img_height = self.params["image_params"]["img_height"]
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
        training_generator = DataGenerator(train_filesname, raw_directory, data_directory, batch_size=batch_size, inp_dim=(288, 512),
                                           des_dim=(288, 512), n_channels=1,
                                           shuffle=True, data_format=data_format, outputs=required_outputs)

        val_data_directory = os.path.join(main_img_dir, "val", "meta_data")
        val_raw_directory = os.path.join(main_img_dir, "val", "front_view_image")
        val_filesname = DataUtils.load_all_file_names(val_raw_directory, required_name_start='front_view_image', img_format="png")

        validation_generator = DataGenerator(val_filesname, val_raw_directory, val_data_directory, batch_size=batch_size, inp_dim=(288, 512),
                                             des_dim=(288, 512), n_channels=1,
                                             shuffle=True, augment=None, data_format=data_format, outputs=required_outputs)

        # temp_ids = train_filesname[:5]
        # training_generator.generate_data_with_list_ids(temp_ids)

        tensorboard = TensorBoard(log_dir=save_path,
                                  write_graph=True, write_images=False)

        logger_tensorboard1 = TrainLogger(os.path.join(save_path, 'train_loss'))
        logger_tensorboard2 = TrainLogger(os.path.join(save_path, 'val_loss'))

        logger_tensorboard3 = TrainLogger(os.path.join(save_path, 'train_horizon'))
        logger_tensorboard4 = TrainLogger(os.path.join(save_path, 'val_horizon'))
        #
        logger_tensorboard5 = TrainLogger(os.path.join(save_path, 'train_scene_class'))
        logger_tensorboard6 = TrainLogger(os.path.join(save_path, 'val_scene_class'))
        model, data_format = self.model_manager.get_model_for_train(model_name, n_labels, input_shape, init_filters_num, train_backbone)
        print(model.summary())
        if self.setup["load_model"]:
            presaved_model = self.setup["model_to_load"]
            model.load_weights(presaved_model, by_name=True)
        self.model_manager.compile_model(model, model_name, self.params['train_params'])

        i_epoch = 1
        dataset_name = main_img_dir.split('\\')[-1]
        filename_init = dataset_name + date_time_str + "_nfltr" + str(init_filters_num) + "run_v_"
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
            logger_tensorboard1.scalar_summary('loss', history.history["loss"][0], i_epoch)
            logger_tensorboard2.scalar_summary('loss', history.history["val_loss"][0], i_epoch)
            logger_tensorboard3.scalar_summary('horizon', history.history["horizon_loss"][0], i_epoch)
            logger_tensorboard4.scalar_summary('horizon', history.history["val_horizon_loss"][0], i_epoch)
            logger_tensorboard5.scalar_summary('scene_class', history.history["scene_class_loss"][0], i_epoch)
            logger_tensorboard6.scalar_summary('scene_class', history.history["val_scene_class_loss"][0], i_epoch)

            # save_path, seg_model, filename, val_loss, val_acc
            filename = filename_init + str(i_epoch)
            print("Saving into: ", save_path)
            save_model_thin(save_path, model, filename, val_loss=history.history["loss"][0], val_acc=0)
            print(" ************** @@@@@@ just fitted generator, i_epoch=", i_epoch)

            i_epoch += 1


if __name__ == "__main__":
    tmh = Train()
    tmh.train()
