import random
import cv2
import os
import numpy as np
# import src.cfgs.colors as colors

IMG_HEIGHT = 288.
IMG_WIDTH = 512.

one_over_img_height = 1.0 / IMG_HEIGHT
one_over_img_width = 1.0 / IMG_WIDTH


def img2hybrid_single_example_batch(img, num_columns=50):
    img_l = list(img)
    img_h_l = img2_hybrid_points(img, num_columns=70)
    return img_h_l[0]


def img2_new_hybrid(img, num_columns=70):

    lanes_img = np.zeros_like(img)
    lanes_img[img == 4] = 4
    lanes_img[img == 3] = 3
    lanes_non_z_img = np.nonzero(lanes_img)

    reduced_img_lanes = np.zeros((img.shape[0], num_columns))
    indices_img_lanes = np.zeros((img.shape[0], num_columns))

    lined_indices = np.zeros((img.shape[0], num_columns))
    half_columns = num_columns // 2

    for i in range(img.shape[0]):
        non_z_lanes_xx = np.nonzero(lanes_non_z_img[0] == i)
        num_col_lanes = min(num_columns, len(non_z_lanes_xx[0]))
        beg_idx_lanes = half_columns - (num_col_lanes // 2)
        end_idx_lanes = beg_idx_lanes + len(lanes_non_z_img[1][non_z_lanes_xx[0][:num_col_lanes]])
        reduced_img_lanes[i, beg_idx_lanes:end_idx_lanes] = img[i, lanes_non_z_img[1][non_z_lanes_xx[0][:num_col_lanes]]]

        indices_img_lanes[i, beg_idx_lanes:end_idx_lanes] = lanes_non_z_img[1][non_z_lanes_xx[0][:num_col_lanes]] / 512.

        lined_indices[i, :] = np.full(num_columns, i / 288.)

    res_lanes = np.concatenate((np.expand_dims(reduced_img_lanes, axis=2), np.expand_dims(indices_img_lanes, axis=2)),
                               axis=2)

    res = np.concatenate((res_lanes, np.expand_dims(lined_indices, axis=2)), axis=2)
    return res

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


def img2points(img, one_over_img_shape=(one_over_img_height, one_over_img_width), max_num_points=None):

    # no_vcls_img = np.zeros_like(img)
    # no_vcls_img[img == 4] = 4
    # no_vcls_img[img == 3] = 3
    non_z_points = np.asarray(np.nonzero(img)).T
    non_z_points_val = np.expand_dims(img[non_z_points[:, -2], non_z_points[:, -1]], axis=1)

    non_z_points = non_z_points * one_over_img_shape

    # points = np.concatenate((non_z_points, non_z_points_val, non_z_points * non_z_points,
    #                          np.expand_dims(non_z_points[:, 0] * non_z_points[:, 1], axis=1)), axis=1)
    points = np.concatenate((non_z_points, non_z_points_val), axis=1)
    if max_num_points is not None:
        # print("num_points", len(points), "max",max_num_points)
        np.random.shuffle(points)
        points = points[:max_num_points]
    return points


def img2points_single_example_batch(img, max_num_points=None):
    non_z_points = np.asarray(np.nonzero(img)).T
    non_z_points_val = np.expand_dims(img[0, non_z_points[:, 1], non_z_points[:, 2], 0], axis=1)

    points = np.concatenate((non_z_points[:,1:3], non_z_points_val), axis=1)
    if max_num_points is not None:
        np.random.shuffle(points)
        points = points[:max_num_points]
    return points

class DataUtils:
    @staticmethod
    def load_all_file_names(directory, img_format, required_name_start='', required_name_end='', filesnames=None):
        all_filesname = []

        for counter, filename in enumerate(os.listdir(directory)):
            if filesnames is not None:
                if filename not in filesnames:
                    continue
            if filename.endswith(required_name_end + ".{:}".format(img_format)) and filename.startswith(
                    required_name_start):
                all_filesname.append(filename.replace(required_name_end, ''))

        return all_filesname

    @staticmethod
    def load_all_pictures_in_directory(directory, img_height, img_width, channels, parameters_for_prepare=None,
                                       img_format="png", prepare_img_action=None, cap=-1, required_name_end='', filesnames=None):

        if filesnames is None:
            all_filesname = DataUtils.load_all_file_names(directory, img_format, required_name_end)
        else:
            all_filesname = filesnames

        if 0 < cap < len(all_filesname):
            filenames = random.sample(all_filesname, cap)
        else:
            filenames = all_filesname

        imgs = np.zeros((len(filenames), 1167, 1167, channels))

        for idx, filename in enumerate(filenames):
            # print("idx, filename", idx, filename)
            if (len(required_name_end) > 0) & ( ('full' in filename) | ('MAN' in filename) ):
                corrected_file_name = filename.replace('.png', required_name_end + ".png")
            elif ".png" in filename:
                corrected_file_name = filename
            else:
                continue

            img = cv2.imread(os.path.join(directory, corrected_file_name))
            # print("directory, corrected_file_name", directory, corrected_file_name)
            if img is None:
                print("for filename", corrected_file_name, "IMAGE IS NONE!")
                continue
            if prepare_img_action is not None:
                img = prepare_img_action(img, img_height, img_width, parameters_for_prepare)
            if not (img.shape[0] == imgs.shape[1]):
                img = cv2.resize(img, (1167, 1167))
            imgs[idx, :, :, :] = img

        return np.array(filenames), imgs

    @staticmethod
    def read_data_from_directory(raw_directory, mask_directory, val_raw_directory, val_mask_directory, img_width, img_height):
        """
        :param raw_directory:
        :param mask_directory:
        :param val_raw_directory:
        :param val_mask_directory:
        :param img_width:
        :param img_height:
        :return:
        """
        # loading training set
        train_file_names, orig_train_x = DataUtils.load_all_pictures_in_directory(raw_directory, img_height, img_width,
                                                                                  parameters_for_prepare=None,
                                                                                  img_format="png", prepare_img_action=None, cap=-1)

        _, orig_train_y = DataUtils.load_all_pictures_in_directory(mask_directory, img_height, img_width, parameters_for_prepare=None,
                                                                   img_format="png", prepare_img_action=None, cap=-1)

        # loading validation set
        val_file_names, orig_val_x = DataUtils.load_all_pictures_in_directory(val_raw_directory, img_height, img_width, parameters_for_prepare=None,
                                                                              img_format="png", prepare_img_action=None, cap=-1)

        _, orig_val_y = DataUtils.load_all_pictures_in_directory(val_mask_directory, img_height, img_width, parameters_for_prepare=None,
                                                                 img_format="png", prepare_img_action=None, cap=-1)

        return orig_train_x, orig_train_y, orig_val_x, orig_val_y

    def get_partition(self, raw_directory, val_raw_directory, img_format):
        train_IDs = self.load_all_file_names(raw_directory, img_format)
        val_IDs = self.load_all_file_names(val_raw_directory, img_format)
        partition = {'train': train_IDs, "validation": val_IDs}
        return partition
