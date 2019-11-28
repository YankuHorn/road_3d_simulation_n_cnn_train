import matplotlib.pyplot as plt
import os
from road_manifold.road_manifold import RoadManifoldPointCloud
from road_top_view_image.tv_image import TV_image
import numpy as np
import cv2
import random
import csv

seg_to_display = [2, 3, 4]
colors = [
    [0, 0, 254],
    [0, 254, 0],
    [254, 0, 0],
    [128, 0, 254],
    [0, 128, 254],
    [254, 128, 0],
    [254, 254, 0],
    [0, 254, 128],
    [128, 254, 0],
    [128, 128, 128],
    [254, 0, 128],
    [128, 0, 128]
]


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
            single_object['left'] = int(float(line_stripped[3])) # - int(line_stripped[6]))
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
            display_img_res[orig_img[:, :, 0] == i] = colors[i]
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
            color = colors[9]
        elif single_object['type'] == 'rear_vehicle':
            color = colors[10]
        else:
            print('no color for type', single_object['type'])
            continue
        display_image = draw_rectangle(display_image, color, single_object['top'], single_object['bottom'], single_object['left'], single_object['right'])
    return display_image


def show_seg_images(seg_dir):
    files_list = os.listdir(seg_dir)

    for j in range(len(files_list)):
        rand_ind = random.randint(0,len(files_list))
        seg_filename = files_list[rand_ind]
        if is_seg_file(seg_filename):
            full_path_seg_image = os.path.join(seg_dir, seg_filename)
            orig_img = cv2.imread(full_path_seg_image)

            display_image = np.zeros((orig_img.shape[0], orig_img.shape[1], 3))
            display_image = add_seg_layer(display_image, orig_img)

            csv_filename = seg_filename.replace('seg', 'out').replace('png', 'csv')
            full_path_csv = os.path.join(seg_dir, csv_filename)
            objects = read_objects_csv(full_path_csv)
            display_image = add_objects_layer(display_image, objects)


            fig = plt.figure(figsize=(2, 1))
            fig.add_subplot(2, 1, 1)
            plt.imshow(display_image)
            raw_filename = seg_filename.replace('seg', 'img')
            full_path_raw_image = os.path.join(seg_dir, raw_filename)
            raw_img = cv2.imread(full_path_raw_image)
            #plt.figure('raw_image')
            fig.add_subplot(2, 1, 2)
            plt.imshow(raw_img)

            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
            # mng = plt.get_current_fig_manager()
            # mng.full_screen_toggle()
            plt.show()

            print('for the b-point')

if __name__ == "__main__":
    seg_dir_name = 'D:\\phantomAI\\data\\collected_data\\2019-05-09-12-19-52_MapTest\\I92_exit_I280\\I92_exit_I280'
    show_seg_images(seg_dir_name)