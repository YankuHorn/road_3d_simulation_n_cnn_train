import numpy as np
import keras
import cv2
import os
from training.data.data_utils import img2points, img2_hybrid_points
import random
# import matplotlib.pyplot as plt
import json
from training.data.data_utils import one_over_img_width, one_over_img_height

img_height_resize_factor = 1208./ 288.
img_height_crop_from_top = 120.
one_over_orig_cropped_for_resize_img_height = 1/1080.


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data_dir, labels_dir, batch_size=10, max_num_columns=50,
                 shuffle=True, augment=None, data_format="channels_last", outputs=None):
        'Initialization'
        self.labels_dir = labels_dir
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.max_num_columns = max_num_columns
        self.shuffle = shuffle
        self.on_epoch_end()
        self.augment = augment
        self.data_dir = data_dir
        self.data_format = data_format
        # self.data_manager = DataUtils()
        self.outputs = outputs

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    @staticmethod
    def read_img(filename, use_flat_32_type, one_channel, flip):
        """
        Wrapper to the cv2 imread func that helps us unify different pics and read options
        """

        if one_channel:
            img = cv2.imread(filename, -1)
        else:
            img = cv2.imread(filename)
        if img is None:
            print('in conv_data_generator.py - read_img function - image is None ; filename=', filename)
            return img
        if use_flat_32_type & (img is not None):
            img = img.astype(np.float32)
        if img.shape[:2] == (288, 512):
            if flip:
                img = cv2.flip(img, 1)
            return img
        else:
            print("something is strange here - input does not follow the normal habbit - please check or cvhange the code according to input size")
            return False

    def _load_data(self, dir_name, ID, flip):

        img = self.read_img(os.path.join(dir_name, ID), use_flat_32_type=False, one_channel=True, flip=flip)

        hybrid_img = img2_hybrid_points(img, self.max_num_columns)

        return hybrid_img

    def load_data_external(self, dir_name, ID):

        return self._load_data(dir_name, ID)

    def load_labels_external(self, dir_name, ID):

        return self._load_labels(dir_name, ID)

    def read_json(self, file_full_path):
        with open(file_full_path) as json_file:
            data = json.load(json_file)
        return data

    def _load_labels(self, dir_name, ID, img_type='seg_resized', flip=False):
        res_dict = dict()

        label_ID = ID.replace("front_view_image", "meta_data")
        label_ID = label_ID.replace(".png", ".json")

        if os.path.isfile(os.path.join(dir_name, label_ID)):
            json_data = self.read_json(os.path.join(dir_name, label_ID))
            horizon_key = None
            host_yaw_at_100m_key = None
            if img_type == 'seg_cropped':
                horizon_key = 'seg_cropped_y_center_host_in_100m'
                host_yaw_at_100m_key = 'seg_cropped_x_center_host_in_100m'
            elif img_type == 'seg_resized':
                horizon_key = 'seg_resized_y_center_host_in_100m'
                host_yaw_at_100m_key = 'seg_resized_x_center_host_in_100m'
            else:
                print("unknown image type name {:}".format(img_type))
            # res_dict['horizon'] = json_data[horizon_key] * one_over_img_height
            # res_dict['horizon'] = json_data[horizon_key] * one_over_img_height
            res_dict['host_yaw_at_100m'] = json_data[host_yaw_at_100m_key] * one_over_img_width
            res_dict['horizon'] = (json_data['front_view_image_horizon'] - 120) * one_over_orig_cropped_for_resize_img_height
            # print("XXX",label_ID,"  ", res_dict['horizon'])
            if json_data['exit_decision'] == 'is_exit':
                res_dict['scene_class'] = [0., 1., 0.]
            elif json_data['merge_decision'] == 'is_merge':
                res_dict['scene_class'] = [0., 0., 1.]
            elif (json_data['exit_decision'] == 'no_exit') and (json_data['merge_decision'] == 'no_merge'):
                res_dict['scene_class'] = [1., 0., 0.]
            elif (json_data['exit_decision'] == 'dont_care') or (json_data['merge_decision'] == 'dont_care'):
                res_dict['scene_class'] = [0.334, 0.333, 0.333]
            else:
                print('no proper exit_decision {:} and merge decision {:}', json_data['exit_decision'], json_data['merge_decision'])

        return res_dict

    @staticmethod
    def init_labels(outputs, num_samples):
        horizon = None
        scene_class = None
        manipulated_horizons = None
        host_yaw_at_100m = None
        if 'horizon' in outputs:
            horizon = np.empty((num_samples), dtype=np.float)
        if 'host_yaw_at_100m' in outputs:
            host_yaw_at_100m = np.empty((num_samples), dtype=np.float)
        if 'scene_class' in outputs:
            scene_class = np.empty((num_samples, 3), dtype=np.float)  # 3 - number of faulty classes
        if 'manipulated_horizons' in outputs:
            manipulated_horizons = np.empty((num_samples, 2), dtype=np.float)  # 2 - number of manipulated images - one resized and one cropped
        return horizon, host_yaw_at_100m, scene_class, manipulated_horizons

    @staticmethod
    def _manipulate_horizon(horizon):
        resized_horizon = horizon * img_height_resize_factor
        cropped_horizon = horizon - img_height_crop_from_top

        return resized_horizon, cropped_horizon

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # max_num_points = self.max_num_points
        X = np.zeros((len(list_IDs_temp), 288, self.max_num_columns, 5), dtype=np.int64)
        horizon, host_yaw_at_100m, scene_class, manipulated_horizons = self.init_labels(self.outputs, len(list_IDs_temp))
        # Generate data
        flip_data = list()
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            flip = bool(random.getrandbits(1))
            file_fn = os.path.join(self.data_dir, ID)
            if not os.path.isfile(file_fn):
                print("something is wrong - this is not a file", file_fn)
            seg_img_ID = ID.replace("front_view_image", "seg_front_view_image")
            hybrid_img = self._load_data(self.data_dir, seg_img_ID, flip)
            # num_points = min(len(points), max_num_points)
            X[i, :] = hybrid_img
            # Store class
            labels = self._load_labels(self.labels_dir, ID)
            # manipulated_horizons[i, ] = self._manipulate_horizon(labels['horizon'])

            if 'horizon' in self.outputs:
                horizon[i, ] = labels['horizon']
            if 'host_yaw_at_100m' in self.outputs:
                host_yaw_at_100m[i, ] = labels['host_yaw_at_100m']
            if 'scene_class' in self.outputs and ('scene_class' in labels.keys()):
                scene_class[i, ] = labels['scene_class']
            if 'manipulated_horizons' in self.outputs:
                manipulated_horizons[i, ] = self._manipulate_horizon(labels['horizon'])
                manipulated_horizons[i, ] = labels['manipulated_horizons']

        aggregated_labels = {'horizon': horizon, 'scene_class': scene_class, 'host_yaw_at_100m': host_yaw_at_100m, 'manipulated_horizons': manipulated_horizons}

        return X, aggregated_labels

    def generate_raw_data_with_list_ids(self, temp_list_ids):

        X, y = self.__data_generation(temp_list_ids)
        return X, y
