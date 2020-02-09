import matplotlib.pyplot as plt
import random
import numpy as np
import time

from lean_inference.tools import get_img2pred_of_collected_data, get_img2show_of_collected_data
from lean_inference.models.architecture.my_inception_v3 import My_inception
from lean_inference.models.architecture.my_conv_net import my_convnet
from lean_inference.models.architecture.my_hybrid_net import my_hybrid_network
from lean_inference.tools import get_collected_data_full_seg_non_cropped_paths_list
from lean_inference.tools import img2_hybrid_points, draw_horizon_line, save_as_jpgs
import cv2

RUN_ON_GPU = True
import os

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
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7

orig_cropped_for_resize_img_height = 1080 - 120


def Infere_model_and_save_result(model_name, images_parent_dir, trgt_dir):
    model = None
    model_full_path = None

    if model_name == "my_conv_net":
        model_full_path = 'models\\weights\\2020_01_28__23_00_nfltr8run_v_334.h5'
    elif model_name == "hybrid":
        model_full_path = 'models\\weights\\2020_01_28__00_40_nfltr8run_v_218.h5'
    elif model_name == "inception_v3":
        model_full_path = 'models\\weights\\2020_01_28__07_02_nfltr8run_v_480.h5'
    else:
        print("name of model not known:", model_name)
    if model_name == "hybrid":
        model = my_hybrid_network()
    elif model_name == "my_conv_net":
        model = my_convnet(input_shape=(288, 512, 1), kernel=3,
                           pool_size=(2, 2), filters_init_num=8,
                           data_format="channels_last")
    elif model_name == 'inception_v3':
        model = My_inception(input_shape=(288, 512, 1), kernel=3, n_labels_scene_class=3,
                             pool_size=(2, 2), filters_init_num=8,
                             data_format="channels_last")

    model.load_weights(model_full_path)

    print(model.summary())

    # all_files = get_collected_data_full_seg_non_cropped_paths_list(images_parent_dir)
    #
    # random.shuffle(all_files)

    pre_process_time_init = None

    if model_name == 'hybrid':
        max_columns = 70
        X = np.zeros((1, 288, max_columns, 5))
    elif (model_name == 'my_conv_net') or (model_name == 'inception_v3'):
        X = np.zeros((1, 288, 512, 1))

    # hack for one clip attempt:
    single_clip_dir = 'D:\\phantomAI\\data\\collected_data\\2019-06-27-14-00-57_Ford_101_92_280\\I92_merge_exit_12C\\I92_merge_exit_12C'
    all_files = os.listdir(single_clip_dir)

    for raw_filename_ in all_files:
        with_painte_gt_filename = raw_filename_.replace('seg', 'with_painte_gt')
        trgt_fn = os.path.join(trgt_dir, with_painte_gt_filename)
        if os.path.isfile(trgt_fn):
            continue
        if '.csv' in raw_filename_:
            continue
        filename_ = os.path.join(single_clip_dir, raw_filename_)
        if ('seg_front_center' in filename_) or ('out_front_center_crop' in filename_):
            image_for_show = get_img2show_of_collected_data(filename_)
            if (model_name == 'my_conv_net') or (model_name == 'inception_v3'):
                img_for_pred = get_img2pred_of_collected_data(filename_)
                X[0, :] = img_for_pred
            if model_name == 'hybrid':
                img_for_pred = get_img2pred_of_collected_data(filename_)
                img_for_pred = img_for_pred.squeeze()
                pre_process_time_init = time.time()
                hybrid_img = img2_hybrid_points(img_for_pred, max_columns)
                pre_process_time_end = time.time()
                X[0, :] = hybrid_img
            start_time = time.time()
            prediction = model.predict(X, verbose=0)
            end_time = time.time()
            if pre_process_time_init is not None:
                print("pre-process time  for 1 image", pre_process_time_end - pre_process_time_init)
            print("prediction time  for 1 image", end_time-start_time)

            horizon_on_full_img = (int((prediction[0]) * 1080) + 120)
            # horizon_on_seg_img = prediction[0] * 288

            print("horizon:", horizon_on_full_img)
            draw_horizon_line(image_for_show, horizon_on_full_img, line_width=10, clr=[255, 0, 0])

            #bwith_painte_gt_filename = with_painte_gt_filename.replace('.png', '.jpg')
            horizon_on_seg_img = prediction[0] * 288

            save_as_jpgs(image_for_show, filename_, '', horizon_on_seg_img, trgt_fn)
            # cv2.imwrite(trgt_fn, image_for_show)

if __name__ == "__main__":
    model_name_ = 'my_conv_net'
    # model_name = 'hybrid'
    # model_name = 'inception_v3'

    images_parent_dir_ = 'D:\\phantomAI\\data\\collected_data\\'
    # output_dir = 'D:\\phantomAI\\data\\results_on_collected\\'
    target_dir_ = 'D:\\phantomAI\\data\\results_on_collected\\2019-06-27-14-00-57_Ford_101_92_280I92_merge_exit_12C\\my_conv_net_2020_01_28__23_00_nfltr8run_v_334'

    Infere_model_and_save_result(model_name_, images_parent_dir_, target_dir_)