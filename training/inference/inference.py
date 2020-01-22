import matplotlib.pyplot as plt
from handle_data.handle_collected_data import read_objects_csv
from tools.draw_tools import type_name2type_idx, type_name2color
from tools.draw_tools import draw_rect, seg_img2clr_img, keep_indices_in_seg_img
import random
#import training.utils as utils
import os
from training.data.data_utils import DataUtils
from training.models_manager.models_manager import ModelManager
from tools.draw_tools import draw_horizon_cross
import numpy as np
import cv2
from training.models_manager.my_losses import my_L2_loss
from training.data.data_utils import img2points_single_example_batch, img2_hybrid_points
from keras import models
from keras.callbacks import TensorBoard
# import tensorflow as tf
import time
import json

orig_cropped_for_resize_img_height = 1080 - 120

def get_collected_data_full_seg_non_cropped_paths_list(parent_dir=None):

    full_paths_seg_not_cropped = list()
    if parent_dir is None:
        parent_dir = 'D:\\phantomAI\\data\\collected_data'
    dirs_in_parent = os.listdir(parent_dir)

    for session_dir in dirs_in_parent:
        session_full_path = os.path.join(parent_dir, session_dir)
        if not os.path.isdir(session_full_path):
            continue
        clip_dirs = os.listdir(session_full_path)
        for clip_dir in clip_dirs:
            clip_full_path = os.path.join(session_full_path, clip_dir, clip_dir) # for some reason...
            if not os.path.isdir(clip_full_path):
                continue
            filenames = os.listdir(clip_full_path)
            for filename in filenames:
                if ('seg_front_center' in filename) and ('seg_front_center_crop' not in filename):
                    full_path_seg = os.path.join(clip_full_path, filename)
                    full_paths_seg_not_cropped.append(full_path_seg)
    return full_paths_seg_not_cropped


def get_img2pred_of_synthsized_data(images_dir, filename):

    img_for_pred = cv2.imread(os.path.join(images_dir, filename))
    img_for_pred = np.expand_dims(img_for_pred, 0)
    img_for_pred = np.expand_dims(img_for_pred[:, :, :, 0], 3)
    return img_for_pred


def getimg2show_of_synthsized_data(images_dir, filename):
    img2show = cv2.imread(os.path.join(images_dir, filename.replace('seg_', "")))
    return img2show

    # image_for_show_seg = cv2.imread(os.path.join(images_dir, filename.replace('se9_for_prediction', "seg")))


def get_img2show_of_collected_data(filename):
    seg_image = cv2.imread(filename)
    csv_file_full_path = filename.replace('seg', 'out').replace('png', 'csv')
    objects = read_objects_csv(csv_file_full_path)

    color_image = seg_img2clr_img(seg_image)
    for single_object in objects:
        # if single_object['type'] == 'Vehicle':
        #     clr = type_name2color('vehicle')
        if single_object['type'] == 'rear_vehicle':
            clr = type_name2color('vehicle')
        else:
            print('no color for type', single_object['type'])
            continue
        # r1, c1, r2, c2, clr, width
        color_image = draw_rect(color_image,
                                single_object['top'], single_object['left'],
                                single_object['bottom'], single_object['right'],
                                clr, width=2)
    color_image = cv2.resize(color_image, (1920, 1280))
    return color_image


def get_img2pred_of_collected_data(filename):
    seg_image = cv2.imread(filename, -1)

    csv_file_full_path = filename.replace('seg', 'out').replace('png', 'csv')
    objects = read_objects_csv(csv_file_full_path)
    for single_object in objects:
        # if single_object['type'] == 'Vehicle':
        #     idx = type_name2type_idx('vehicle')
        if (single_object['type'] == 'rear_vehicle') or (single_object['type'] == 'Vehicle'):
            idx = type_name2type_idx('vehicle')
        else:
            print('no color for type', single_object['type'])
            continue
        # r1, c1, r2, c2, clr, width
        seg_image = draw_rect(seg_image,
                                  single_object['top'], single_object['left'],
                                  single_object['bottom'], single_object['right'],
                                  idx, width=2)
    relevant_indices_img = keep_indices_in_seg_img(seg_image, [3, 4, 8])
    return np.expand_dims(np.expand_dims(relevant_indices_img, axis=0), axis=3)

if __name__ == "__main__":
    model_manager = ModelManager()


    n_lables = 3
    input_shape = (288, 512, 1)

    # model = model_manager.get_model_for_inference(model_name, n_lables, input_shape, 8)
    # model_name = "horizon_exit_merge"
    # # model_full_path = 'D:\\phantomAI\\models\\chosen\\2020_01_05__23_43_nfltr8run_v_34.h5'
    # model_full_path = 'D:\\phantomAI\\results\pointnet_cls\\train_2020_01_14__22_31\\2020_01_14__22_31_nfltr16run_v_27.h5'

    # model_name = "hybrid"  # D:\phantomAI\results\hybrid\train_2020_01_15__22_02
    # model_full_path = 'D:\\phantomAI\\results\\hybrid\\train_2020_01_15__23_25\\2020_01_15__23_25_nfltr8run_v_14.h5'

    model_name = "horizon_exit_merge"
    # model_full_path = 'D:\\phantomAI\\results\\horizon_exit_merge\\train_2020_01_14__22_31\\2020_01_14__22_31_nfltr8run_v_53.h5'
    # model_full_path = 'D:\\phantomAI\\results\\horizon_exit_merge\\train_2020_01_20__22_51\\2020_01_20__22_51_nfltr8run_v_28.h5'
    # BEST UNTILL NOW: model_full_path = 'D:\\phantomAI\\results\\horizon_exit_merge\\train_2019_12_24__09_43\\2019_12_24__09_43_nfltr8run_v_151.h5'
    model_full_path = 'D:\\phantomAI\\results\\horizon_exit_merge\\train_2020_01_21__20_37\\2020_01_21__20_37_nfltr8run_v_80.h5'

    from training.models_manager.models.my_pointnet_horizon_HPInc import OrthogonalRegularizer
    # model = models.load_model(model_full_path, custom_objects={'my_L2_loss': my_L2_loss})
    # model = models.load_model(model_full_path,
    #                           custom_objects={'my_L2_loss': my_L2_loss, 'OrthogonalRegularizer': OrthogonalRegularizer})
    if model_name == 'pointnet':
        model = model_manager.get_model_for_inference('pointnet_cls', input_shape=(4096, 4))
    elif model_name == "hybrid":
        model = model_manager.get_model_for_inference('hybrid', input_shape=(288, 50, 4))
    else:
        model, _ = model_manager.get_model_for_train('horizon_exit_merge', input_shape=(288, 512, 1), n_labels=1, init_filters_num=8, train_backbone=False)

    model.load_weights(model_full_path)
    print(model.summary())
    # images_dir = 'C:\\work\\tmp\\I92_merge_exit_9B_merge'
    # images_dir = 'D:\\phantom\\data\\synthesized_data\\2019_12_11__20_32_run_\\front_view_image'

    # images_dir = 'D:\\phantomAI\\data\\synthesized_data\\testing\\2019_12_11__20_32_run_\\train\\front_view_image'
    images_dir = 'D:\\phantomAI\\data\\synthesized_data\\2020_01_06__14_23_run_mixed_with_0107_2031\\val\\front_view_image'
    all_files = os.listdir(images_dir)
    # images_dir = None
    # all_files = get_collected_data_full_seg_non_cropped_paths_list()
    random.shuffle(all_files)

    # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    # model.fit(x_train, y_train, verbose=1, callbacks=[tensorboard])
    max_num_points = 4096
    max_columns = 50
    # X = np.zeros((1, max_num_points, 3), dtype=np.int64)
    pre_process_time_init = None

    if model_name == 'pointnet':
        X = np.zeros((1, 4096, 3))
    elif model_name == 'hybrid':
        X = np.zeros((1, 288, 50, 5))
    elif model_name == 'horizon_exit_merge':
        X = np.zeros((1, 288, 512, 1))
    for filename_ in all_files:

        if 'seg_front_view_image' in filename_:
            meta_data_dir = images_dir.replace('front_view_image', 'meta_data')
            meta_data_file_name = (filename_.replace('seg_front_view_image', 'meta_data')).replace('.png', '.json')
            json_fp = os.path.join(meta_data_dir, meta_data_file_name)
            with open(json_fp) as json_file:
                data = json.load(json_file)
            gt_horizon = data['seg_resized_y_center_host_in_100m']
            if images_dir is not None:
                image_for_show = getimg2show_of_synthsized_data(images_dir, filename_)
                if model_name == 'horizon_exit_merge':
                    img_for_pred = get_img2pred_of_synthsized_data(images_dir, filename_)
                    X[0, ] = img_for_pred # np.transpose(img_for_pred, (0, 2, 1))

                    # X = np.flip(np.transpose(img_for_pred, (0, 2, 1, 3)))
                if model_name == 'pointnet':
                    img_for_pred = get_img2pred_of_synthsized_data(images_dir, filename_)
                    points = img2points_single_example_batch(img_for_pred, max_num_points)
                    num_points = min(len(points), max_num_points)
                    X[0, :num_points] = points[:num_points]
                elif model_name == 'hybrid':
                    # img = self.read_img(os.path.join(dir_name, ID), use_flat_32_type=False, one_channel=True, flip=flip)

                    img = cv2.imread(os.path.join(images_dir, filename_), -1)
                    # hybrid_img = img2_hybrid_points(img, self.max_num_columns)
                    pre_process_time_init = time.time()
                    hybrid_img = img2_hybrid_points(img, max_columns)
                    pre_process_time_end = time.time()
                    # num_points = min(len(points), max_num_points)
                    X[0, :] = hybrid_img

            else:
                img_for_pred = get_img2pred_of_collected_data(filename_)
                image_for_show = get_img2show_of_collected_data(filename_)
                # image_for_show = seg_img2clr_img(img_for_pred[0, :, :, 0])
            start_time = time.time()
            prediction = model.predict(X, verbose=0)
            end_time = time.time()
            if pre_process_time_init is not None:
                print("pre-process time  for 1 image", pre_process_time_end - pre_process_time_init)
            print("prediction time  for 1 image", end_time-start_time)

            from training.data.conv_data_generator import img_height_resize_factor
            # correct way:
            horizon_on_full_img = (int((prediction[0]) * 1080) + 120)
            horizon_on_seg_img = prediction[0] * 288
            # wrong temp way:
            # horizon = int((prediction[0] * 1280)[0][0])
            chosen_idx = np.argmax(prediction[1])
            classes = {0: 'NORMAL', 1: 'EXIT', 2: 'MERGE'}
            print("horizon:", horizon_on_full_img)
            print(" GT HORIZON on seg imag", gt_horizon, "pred horizon on seg img", horizon_on_seg_img, "DIFF: ", gt_horizon - horizon_on_seg_img)
            # print("scene class probs", prediction[1], "chosen class is ", classes[chosen_idx])
            draw_horizon_cross(image_for_show, 960, horizon_on_full_img, cam_roll=0, clr=[255, 0, 0])
            plt.imshow(image_for_show)
            # mng = plt.get_current_fig_manager()
            # mng.resize(*mng.window.maxsize())
            plt.show()
            print("for b point")
