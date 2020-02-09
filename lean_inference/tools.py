import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path

def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent

def keep_indices_in_seg_img(seg_img, indices_list):

    img2keep = np.zeros(shape=seg_img.shape, dtype=np.uint8)
    for obj_idx in indices_list:
        img_indices = np.where(seg_img[:, :] == obj_idx)
        img2keep[img_indices[0], img_indices[1]] = obj_idx

    return img2keep


def seg_img2clr_img(seg_img):

    clr_image = np.zeros(shape=(seg_img.shape[0], seg_img.shape[1], 3), dtype=np.uint8)

    solid_indices = np.where(seg_img[:, :] == type_name2type_idx('solid'))
    dashed_indices = np.where(seg_img[:, :] == type_name2type_idx('dashed'))
    vcl_indices = np.where(seg_img[:, :] == type_name2type_idx('vehicle'))
    clr_image[solid_indices[0], solid_indices[1]] = type_name2color('solid')
    clr_image[dashed_indices[0], dashed_indices[1]] = type_name2color('dashed')
    clr_image[vcl_indices[0], vcl_indices[1]] = type_name2color('vehicle')
    return clr_image


def swap(x1, x2):
    return x2, x1


def draw_line2(img, r1, c1, r2, c2, clr, width=5):

    if abs(r1 - r2) > abs(c1 - c2):

        # a = (c1-c2)/(float(r1-r2))
        # b = c1 - a * r1
        if r1 > r2:
            r1, r2 = swap(r1, r2)
        for r in range(r1, r2):
            c = int((c1-c2)/(float(r1-r2)) * r + c1 - ((c1-c2)/(float(r1-r2))) * r1)
            if (r > 0) and (c > 0) and (r < img.shape[0]) and (c < img.shape[1]):
                img[r, c - width // 2: c + width // 2] = clr
    else:
        # a = (r1-r2)/(float(c1-c2))
        # b = c1 - a * r1
        if c1 > c2:
            c1, c2 = swap(c1, c2)
        for c in range(c1, c2):
            r = int((r1 - r2) / (float(c1 - c2)) * c + r1 - ((r1 - r2) / (float(c1 - c2))) * c1)
            left = r - width // 2
            if (r > 0) and (c >= 0) and (r < img.shape[0]) and (c < img.shape[1]):
                img[r - width // 2: r + width // 2, c] = clr
            # img[r, c] = clr
    return img


def draw_rect(img, r1, c1, r2, c2, clr, fill_clr=None, width=15):

    img = draw_line2(img, r1, c1, r1, c2, clr, width)
    img = draw_line2(img, r1, c1, r2, c1, clr, width)

    img = draw_line2(img, r1, c2, r2, c2, clr, width)
    img = draw_line2(img, r2, c1, r2, c2, clr, width)
    if fill_clr is not None:
        if r1 > r2:
            r1, r2 = swap(r1, r2)
        if c1 > c2:
            c1, c2 = swap(c1, c2)
        img[r1+width//2:r2-width//2, c1+width//2:c2-width//2] = fill_clr
    return img


def type_name2color(type_input):
    """
    # pay attention - the first channel ('R', indexed 0) should be unique per type!!!
    """

    color = [0, 0, 0]
    if type_input == 'solid':
        color = [1, 255, 255]
    elif type_input == 'dashed':
        color = [253, 255, 0]
    elif type_input == 'vehicle':
        color = [252, 0, 255]
    else:
        # print("no color for this type yet:", type_input)
        pass
    return color


def type_name2type_idx(type_input):
    """
    # pay attention - the first channel ('R', indexed 0) should be unique per type!!!
    """
    type_idx = None
    if type_input == 'solid':
        type_idx = 3
    elif type_input == 'dashed':
        type_idx = 4
    elif type_input == 'vehicle':
        type_idx = 8
    else:
        # print("no color for this type yet:", type_input)
        pass
    return type_idx


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


def get_collected_data_full_seg_non_cropped_paths_list(parent_dir):

    full_paths_seg_not_cropped = list()

    dirs_in_parent = os.listdir(parent_dir)

    for session_dir in dirs_in_parent:
        session_full_path = os.path.join(parent_dir, session_dir)
        if not os.path.isdir(session_full_path):
            continue
        clip_dirs = os.listdir(session_full_path)
        for clip_dir in clip_dirs:
            clip_full_path = os.path.join(session_full_path, clip_dir, clip_dir)  # second clip_dir - for some reason...
            if not os.path.isdir(clip_full_path):
                continue
            filenames = os.listdir(clip_full_path)
            for filename in filenames:
                if 'seg_front_center' in filename:
                    full_path_seg = os.path.join(clip_full_path, filename)
                    full_paths_seg_not_cropped.append(full_path_seg)
    return full_paths_seg_not_cropped


def get_img2show_of_collected_data(filename):
    seg_image = cv2.imread(filename)
    csv_file_full_path = filename.replace('seg', 'out').replace('png', 'csv')
    objects = read_objects_csv(csv_file_full_path)

    color_image = seg_img2clr_img(seg_image)
    for single_object in objects:
        if single_object['type'] == 'rear_vehicle':
            clr = type_name2color('vehicle')
        else:
            # print('no color for type', single_object['type'])
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
        if (single_object['type'] == 'rear_vehicle') or (single_object['type'] == 'Vehicle'):
            idx = type_name2type_idx('vehicle')
        else:
            # print('no color for type', single_object['type'])
            continue
        # r1, c1, r2, c2, clr, width
        seg_image = draw_rect(seg_image, single_object['top'], single_object['left'], single_object['bottom'],
                              single_object['right'], idx, width=2)
    relevant_indices_img = keep_indices_in_seg_img(seg_image, [3, 4, 8])
    return np.expand_dims(np.expand_dims(relevant_indices_img, axis=0), axis=3)


def read_ground_truth(images_dir, filename, image_source):
    if image_source is 'collected':
        return None
    meta_data_dir = images_dir.replace('front_view_image', 'meta_data')
    meta_data_file_name = (filename.replace('seg_front_view_image', 'meta_data')).replace('.png', '.json')
    json_fp = os.path.join(meta_data_dir, meta_data_file_name)
    with open(json_fp) as json_file:
        data = json.load(json_file)
    gt_horizon = data['seg_resized_y_center_host_in_100m']
    return gt_horizon


def show_in_plt(display_image, seg_filename, seg_dir, horizon_on_raw):

    fig = plt.figure(figsize=(2, 1))
    fig.add_subplot(2, 1, 1)
    plt.imshow(display_image)
    raw_filename = seg_filename.replace('seg', 'img')
    full_path_raw_image = os.path.join(seg_dir, raw_filename)
    raw_img = cv2.imread(full_path_raw_image)
    # img, horizon, line_width, clr=[0, 0, 255]
    draw_horizon_line(raw_img, horizon_on_raw, line_width=2, clr=[255, 0, 0])
    fig.add_subplot(2, 1, 2)
    plt.imshow(raw_img)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


def save_as_jpgs(display_image, seg_filename, seg_dir, horizon_on_raw, trgt_fn):

    fig = plt.figure(figsize=(12.0, 7.5))
    fig.add_subplot(2, 1, 2)
    plt.imshow(display_image)
    raw_filename = seg_filename.replace('seg', 'img')
    # raw_filename = raw_filename.replace('.png', '.jpg')
    full_path_raw_image = os.path.join(seg_dir, raw_filename)
    raw_img = cv2.imread(full_path_raw_image)
    # img, horizon, line_width, clr=[0, 0, 255]
    draw_horizon_line(raw_img, horizon_on_raw, line_width=2, clr=[255, 0, 0])
    fig.add_subplot(2, 1, 1)
    plt.imshow(raw_img)


    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.savefig(trgt_fn)
    plt.close()

def img2_hybrid_points(img, num_columns=70, num_channels=3):

    lanes_img = np.zeros_like(img)
    lanes_img[img == 4] = 4
    lanes_img[img == 3] = 3
    lanes_non_z_img = np.nonzero(lanes_img)

    vcls_img = np.zeros_like(img)
    vcls_img[img == 8] = 8
    vcls_non_z_img = np.nonzero(vcls_img)

    reduced_img_lanes = np.zeros((img.shape[0], num_columns))
    indices_img_lanes = np.zeros((img.shape[0], num_columns))

    reduced_img_vcls = np.zeros((img.shape[0], num_columns))
    indices_img_vcls = np.zeros((img.shape[0], num_columns))

    lined_indices =  np.zeros((img.shape[0], num_columns))
    half_columns = num_columns // 2

    for i in range(img.shape[0]):
        non_z_lanes_xx = np.nonzero(lanes_non_z_img[0] == i)
        non_z_vcls_xx = np.nonzero(vcls_non_z_img[0] == i)
        num_col_lanes = min(num_columns, len(non_z_lanes_xx[0]))
        num_col_vcls = min(num_columns, len(non_z_vcls_xx[0]))
        beg_idx_lanes = half_columns - (num_col_lanes // 2)
        end_idx_lanes = beg_idx_lanes + len(lanes_non_z_img[1][non_z_lanes_xx[0][:num_col_lanes]])
        beg_idx_vcls = half_columns - (num_col_vcls // 2)
        end_idx_vcls = beg_idx_vcls + len(vcls_non_z_img[1][non_z_vcls_xx[0][:num_col_vcls]])
        reduced_img_lanes[i, beg_idx_lanes:end_idx_lanes] = img[i, lanes_non_z_img[1][non_z_lanes_xx[0][:num_col_lanes]]]
        reduced_img_vcls[i, beg_idx_vcls:end_idx_vcls] = img[i, vcls_non_z_img[1][non_z_vcls_xx[0][:num_col_vcls]]]

        indices_img_lanes[i, beg_idx_lanes:end_idx_lanes] = lanes_non_z_img[1][non_z_lanes_xx[0][:num_col_lanes]] / 512.
        indices_img_vcls[i, beg_idx_vcls:end_idx_vcls] = vcls_non_z_img[1][non_z_vcls_xx[0][:num_col_vcls]] / 512.

        lined_indices[i, :] = np.full(num_columns, i / 288.)

    res_lanes = np.concatenate((np.expand_dims(reduced_img_lanes, axis=2), np.expand_dims(indices_img_lanes, axis=2)),
                               axis=2)
    res_vcls = np.concatenate((np.expand_dims(reduced_img_vcls, axis=2), np.expand_dims(indices_img_vcls, axis=2)),
                               axis=2)
    res = np.concatenate((res_lanes, res_vcls, np.expand_dims(lined_indices, axis=2)), axis=2)
    return res


def draw_horizon_line(img, horizon, line_width, clr=[0, 0, 255]):

    width = img.shape[1]
    draw_line2(img, horizon, 0,
               horizon, width, clr=clr, width=line_width)
    return img
