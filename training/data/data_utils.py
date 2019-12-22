import random
import cv2
import os
import numpy as np
# import src.cfgs.colors as colors


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
