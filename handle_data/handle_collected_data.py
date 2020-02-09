import matplotlib.pyplot as plt
import os
from road_manifold.road_manifold import RoadManifoldPointCloud
from road_top_view_image.tv_image import TV_image
import numpy as np
import cv2
import random
import csv
from tools.draw_tools import type_name2color, idx2type, type2idx


seg_to_display = [3, 4]


def is_seg_file(filename):
    return filename.startswith('seg')


def read_objects_csv(filename):
    f = open(filename, "r")
    lines = f.read().split('\n')
    objects = list()
    for line in lines[1:-1]:
        line_stripped = [x.strip() for x in line.split(',')]
        if line_stripped[2] == '1':
            single_object = dict()
            single_object['type'] = line_stripped[1]
            single_object['bottom'] = int(float(line_stripped[4])) + int(float(line_stripped[6]))
            single_object['top'] = int(float(line_stripped[4]))
            single_object['left'] = int(float(line_stripped[3]))  # - int(line_stripped[6]))
            single_object['right'] = int(float(line_stripped[3])) + int(float(line_stripped[5]))
            objects.append(single_object)
        if line_stripped[7] == '1':
            single_object_rear = dict()
            single_object_rear['type'] = 'rear_vehicle'
            single_object_rear['bottom'] = int(float(line_stripped[9])) + int(float(line_stripped[11]))
            single_object_rear['top'] = int(float(line_stripped[9]))  # - int(line_stripped[6])
            single_object_rear['left'] = int(float(line_stripped[8]))  # - int(line_stripped[6])
            single_object_rear['right'] = int(float(line_stripped[8])) + int(float(line_stripped[10]))
            objects.append(single_object_rear)
    return objects


def add_seg_layer(display_img, orig_img):
    display_img_res = display_img.copy()
    for i in range(12):
        if i in seg_to_display:
            display_img_res[orig_img[:, :, 0] == i] = type_name2color(idx2type(i))
    return display_img_res


def draw_rectangle(display_image, color, top, bottom, left, right):
    display_image_res = display_image.copy()
    display_image_res[top:bottom, (left - 1):(left + 1), :] = color
    display_image_res[top:bottom, (right - 1):(right + 1), :] = color
    display_image_res[(top - 1):(top + 1), left:right, :] = color
    display_image_res[(bottom - 1):(bottom + 1), left:right, :] = color
    return display_image_res


def add_objects_layer(display_image, objects):
    for single_object in objects:
        if single_object['type'] == 'Vehicle':
            color = type_name2color('vehicle')

        elif single_object['type'] == 'rear_vehicle':
            color = type_name2color('vehicle')
        else:
            print('no color for type', single_object['type'])
            continue
        display_image = draw_rectangle(display_image, color, single_object['top'], single_object['bottom'],
                                       single_object['left'], single_object['right'])
    return display_image


def save_as_seg_image(display_image, save_path):
    seg_image_full = np.zeros(shape=(display_image.shape[0], display_image.shape[1]), dtype=np.uint8)

    solid_indices = np.where(display_image[:, :, 0] == type_name2color('solid')[0])
    dashed_indices = np.where(display_image[:, :, 0] == type_name2color('dashed')[0])
    vcl_indices = np.where(display_image[:, :, 0] == type_name2color('vehicle')[0])
    seg_image_full[solid_indices[0], solid_indices[1]] = type2idx('solid')
    seg_image_full[dashed_indices[0], dashed_indices[1]] = type2idx('dashed')
    seg_image_full[vcl_indices[0], vcl_indices[1]] = type2idx('vehicle')

    cv2.imwrite(save_path, seg_image_full)

def show_seg_images(seg_dir, show=False, save=False):
    files_list = os.listdir(seg_dir)

    for j in range(len(files_list)):
        # rand_ind = random.randint(0, len(files_list))
        # seg_filename = files_list[rand_ind]
        seg_filename = files_list[j]
        if is_seg_file(seg_filename):
            full_path_seg_image = os.path.join(seg_dir, seg_filename)
            # if 'crop' in seg_filename:
            #     cropped_orig_img = cv2.imread(full_path_seg_image)
            #     full_orig_img = cv2.imread(full_path_seg_image.replace('_crop_', '_'))
            # else:
            #     cropped_orig_img = cv2.imread(full_path_seg_image.replace('center_','center_crop_'))
            full_orig_img = cv2.imread(full_path_seg_image)

            display_image = np.zeros((full_orig_img.shape[0], full_orig_img.shape[1], 3))
            display_image = add_seg_layer(display_image, full_orig_img)

            csv_filename = seg_filename.replace('seg', 'out').replace('png', 'csv')
            full_path_csv = os.path.join(seg_dir, csv_filename)
            objects = read_objects_csv(full_path_csv)
            display_image = add_objects_layer(display_image, objects)
            show_in_plt(display_image, seg_filename, seg_dir)


            print('for the b-point')
            save_path = os.path.join(seg_dir, seg_filename.replace("seg","se9_for_prediction"))
            save_as_seg_image(display_image, save_path)


if __name__ == "__main__":
    # seg_dir_name = 'D:\\phantomAI\\data\\collected_data\\2019-05-09-12-19-52_MapTest\\I92_exit_I280\\I92_exit_I280'
    seg_dir_name = 'D:\\phantomAI\\data\\collected_data\\2019-06-27-14-00-57_Ford_101_92_280\I92_merge_exit_9B_merge'


    show_seg_images(seg_dir_name, show=True, save=True)
