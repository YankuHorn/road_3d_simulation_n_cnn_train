import matplotlib.pyplot as plt
import os
from road_manifold.road_manifold import RoadManifoldPointCloud
from road_manifold.linear_road_model import LinearRoadModel
from road_top_view_image.tv_image import TV_image
import cv2
from datetime import datetime

from tools.draw_tools import draw_rect, draw_line2, draw_horizon_cross
from tools.draw_tools import type_name2color, type_name2type_idx
import numpy as np
dX_threshold = 0.3
dY_threshold = 0.3

epsilon = 0.000000001


# duplicated - also in lanes.py - pay attention if you change !!!
SEG_IMG_DIMS = (512, 288)
CROP_FULL_IMG = (0, 120, 1920, 1080)
CROP_CROPPED_IMG = (704, 516, 512, 288)
resize_factor = (CROP_FULL_IMG[2] / SEG_IMG_DIMS[0], CROP_FULL_IMG[3] / SEG_IMG_DIMS[0])

# duplicated - also in lanes.py - pay attention if you change !!!

class FV_image:
    def __init__(self, width_pix, height_pix, focal_length, camera_height, x_center, y_center, cam_roll=0):
        self.img = np.zeros((height_pix, width_pix, 3), dtype=np.int16)
        self.img_w = width_pix
        self.img_h = height_pix
        self.camH = camera_height
        self.fl = focal_length

        self.one_over_img_w = 1.0 / self.img_w
        self.one_over_img_h = 1.0 / self.img_h
        self.x_center = x_center
        self.y_center = y_center

        self.vehicles = list()

        # self.vcls_lines = list()
        self.cam_roll = cam_roll

        # self.exit_points = list()
        # self.merge_points = list()
        # self.drawed_points = list()
        # self.exit_decision = "no_exit"
        # self.merge_decision = "no_merge"
        # for dumping later
        self.vcl_centers_x = list()
        self.vcl_centers_bottom_y = list()
        self.vcl_widths = list()
        self.vcl_heights = list()
    @staticmethod
    def get_num_indication_lines(points_list):
        indication_lines = list()
        for point in points_list:
            if point[0] not in indication_lines:
                indication_lines.append(point[0])
        num_indication_lines = len(indication_lines)
        return num_indication_lines

    def calc_exit_merge(self):

        # num_exit_indication_lines = self.get_num_indication_lines(self.exit_points)
        # num_merge_indication_lines = self.get_num_indication_lines(self.merge_points)
        num_exit_indication_points = len(self.exit_points)
        num_merge_indication_points = len(self.merge_points)
        if num_exit_indication_points > 80:
            self.exit_decision = "is_exit"
        elif num_exit_indication_points > 10:
            self.exit_decision = "dont_care"
        if num_merge_indication_points > 80:
            self.merge_decision = "is_merge"
        elif num_merge_indication_points > 10:
            self.merge_decision = "dont_care"

    def XYZ2xy(self, X, Y, Z):
        """
        (y - y_center) / focal_length = Y / Z
        (x - x_center) / focal_length = X / Z
         = >
         y = (Y / Z) * focal_length + y_center
         x = (X / Z) * focal_length + x_center
        """
        # if height is None:
        #     height = self.camH

        y = int(round((Y / Z) * self.fl + self.y_center))
        x = int(round((X / Z) * self.fl + self.x_center))
        if (x > 0) and (y > 0) and (x < self.img_w) and (y < self.img_h):
            return x, y
        else:
            return None, None

    def XZ2xy(self, X, Z):
        """
        (y - y_center) / focal_length = Y / Z
        (x - x_center) / focal_length = X / Z
         = >
         y = (Y / Z) * focal_length + y_center
         x = (X / Z) * focal_length + x_center
        """
        # if height is None:
        #     height = self.camH
        Y = - X * np.math.sin(self.cam_roll)
        y = int(round((Y / Z) * self.fl + self.y_center))
        x = int(round((X / Z) * self.fl + self.x_center))
        if (x > 0) and (y > 0) and (x < self.img_w) and (y < self.img_h):
            return x, y
        else:
            return None, None

    def XZ2xy_subpixel(self, X, Z):
        """
        (y - y_center) / focal_length = Y / Z
        (x - x_center) / focal_length = X / Z
         = >
         y = (Y / Z) * focal_length + y_center
         x = (X / Z) * focal_length + x_center
        """
        # if height is None:
        #     height = self.camH
        Y = - X * np.math.sin(self.cam_roll)
        y = np.float16((Y / Z) * self.fl + self.y_center)
        x = np.float16((X / Z) * self.fl + self.x_center)
        if (x > 0) and (y > 0) and (x < self.img_w) and (y < self.img_h):
            return x, y
        else:
            return None, None

    def remove_exit_points(self, x_l, x_r, y_t, y_b):
        for e_point in self.exit_points:
            if (e_point[1] > x_l) and (e_point[1] < x_r) and (e_point[0] > y_t) and (e_point[0] < y_b):
                self.exit_points.remove(e_point)

    def remove_merge_points(self, x_l, x_r, y_t, y_b):

        for m_point in self.merge_points:
            if (m_point[1] > x_l) and (m_point[1] < x_r) and (m_point[0] > y_t) and (m_point[0] < y_b):
                    self.merge_points.remove(m_point)

    def draw_single_vehicle(self, vcl, X, Z, vcl_yaw_angle):
        visible_width = vcl.size['w'] * np.math.cos(vcl_yaw_angle)

        X_l = X + vcl.position_in_lane - 0.5 * visible_width
        X_r = X + vcl.position_in_lane + 0.5 * visible_width
        Z = Z

        # y_ = int(round((height / Z) * self.fl + self.y_center))
        # x_ = int(round((X / Z) * self.fl + self.x_center))
        #
        # y_ = int(round((height / Z) * self.fl + self.y_center))
        # x_ = int(round((X / Z) * self.fl + self.x_center))
        #
        # [x, y] = np.dot([x_, y_], roll_matrix).astype(int)
        vcl_height = vcl.size['h']

        Y_bottom = self.camH - X * np.math.sin(self.cam_roll)
        top_of_vehicle_height = Y_bottom - vcl_height
        # l = left ; r = right ; b = bottom ; t = top
        x_lb, y_lb = self.XYZ2xy(X_l, Y_bottom, Z)
        x_rb, y_rb = self.XYZ2xy(X_r, Y_bottom, Z)
        x_lt, y_lt = self.XYZ2xy(X_l, Y=top_of_vehicle_height, Z=Z)
        x_rt, y_rt = self.XYZ2xy(X_r, Y=top_of_vehicle_height, Z=Z)
        # if (x_lb is None) or (x_rb is None) or (x_lt is None) or (x_rt is None):
        #     print("something is none")
        # if (y_lb is None) or (y_rb is None) or (y_lt is None) or (y_rt is None):
        #     print("something is none")
        # save for meta-data:

        color = type_name2color('vehicle')
        if (x_lb is not None) and (x_rb is not None) and (x_lt is not None) and (x_rt is not None):
            self.img = draw_rect(self.img, y_lb, x_lb, y_rt, x_rt, color, fill_clr=[0, 0, 0], width=8)
            # self.remove_exit_points(x_lb, x_rb, y_lt, y_lb)
            # self.remove_merge_points(x_lb, x_rb, y_lt, y_lb)

            left_av = np.mean([x_lb, x_lt])
            right_av = np.mean([x_rb, x_rt])
            top_av = np.mean([y_lt, y_rt])
            bottom_av = np.mean([y_rb, y_lb])
            self.vcl_centers_x.append((x_lb + x_rb + x_lt + x_rt) * self.one_over_img_w * 0.25)
            self.vcl_centers_bottom_y.append(bottom_av * self.one_over_img_h)
            self.vcl_widths.append((right_av - left_av) * self.one_over_img_w)
            self.vcl_heights.append((bottom_av - top_av) * self.one_over_img_h)

        # ['#', 'ID', 'class_name', 'box_valid', 'x', 'y', 'width', 'height', 'rear_valid', 'x', 'y', 'width', 'height']
        # vcl_line = [vcl.ID, 'vehicle', 0, 0 , 0 ,0, 0, 1, x_lt, y_lt, visible_width, vcl_height]
        # self.vcls_lines.append(vcl_line)

    def draw_exit(self):
        if self.exit_decision == "is_exit":
            cv2.circle(self.img, (250, 90), 30, [0, 0, 255], -1)
        elif self.exit_decision == 'dont_care':
            cv2.circle(self.img, (250, 90), 30, [0, 128, 255], -1)
        else:
            return
        # E:
        draw_line2(self.img, 10, 10, 70, 10, [255,255,255], 3)
        draw_line2(self.img, 10, 10, 10, 30, [255, 255, 255], 3)
        draw_line2(self.img, 40, 10, 40, 30, [255, 255, 255], 3)
        draw_line2(self.img, 70, 10, 70, 30, [255, 255, 255], 3)
        # X:
        draw_line2(self.img, 10, 50, 70, 80, [255, 255, 255], 3)
        draw_line2(self.img, 70, 50, 10, 80, [255, 255, 255], 3)
        # I :
        draw_line2(self.img, 10, 100, 70, 100, [255, 255, 255], 3)
        # T:
        draw_line2(self.img, 10, 130, 70, 130, [255, 255, 255], 3)
        draw_line2(self.img, 10, 115, 10, 145, [255, 255, 255], 3)

    def draw_merge(self):
        if self.merge_decision == "is_merge":
            cv2.circle(self.img, (250, 50), 30, [255, 0, 0], -1)
        elif self.merge_decision == 'dont_care':
            cv2.circle(self.img, (250, 50), 30, [255, 0, 255], -1)
        else:
            return

        # M:
        draw_line2(self.img, 10, 10, 70, 10, [255,255,255], 3)
        draw_line2(self.img, 10, 10, 40, 20, [255, 255, 255], 3)
        draw_line2(self.img, 40, 20, 10, 30, [255, 255, 255], 3)
        draw_line2(self.img, 10, 30, 70, 30, [255, 255, 255], 3)
        # E:
        draw_line2(self.img, 10, 50, 70, 50, [255, 255, 255], 3)
        draw_line2(self.img, 10, 50, 10, 70, [255, 255, 255], 3)
        draw_line2(self.img, 40, 50, 40, 70, [255, 255, 255], 3)
        draw_line2(self.img, 70, 50, 70, 70, [255, 255, 255], 3)

    def draw_horizon_cross(self, FV_point_center_host_in_100m=None):
        x, y = self.x_center, self.y_center
        if FV_point_center_host_in_100m is not None:
            x, y = FV_point_center_host_in_100m
        draw_horizon_cross(self.img, x, y, cam_roll=self.cam_roll, clr=[0, 0, 255])
        # left_of_x_center_r = round(self.y_center - 30 * np.math.sin(self.cam_roll))
        # left_of_x_center_c = round(self.x_center - 30)
        # right_of_x_center_c = round(self.x_center + 30)
        # right_of_x_center_r = round(self.y_center + 30 * np.math.sin(self.cam_roll))
        # draw_line2(self.img, left_of_x_center_r, left_of_x_center_c,
        #            right_of_x_center_r, right_of_x_center_c, clr=[0, 0, 255], width=2)
        # top_y_center_r = round(self.y_center + 10)
        # top_y_center_c = round(self.x_center)
        # bot_y_center_r = round(self.y_center - 10)
        # bot_y_center_c = round(self.x_center)
        # draw_line2(self.img, top_y_center_r, top_y_center_c,
        #            bot_y_center_r, bot_y_center_c, clr=[0, 0, 255], width=2)

    def draw_vehicles(self, vehicles, lane_models):
        v_cntr = 0
        # sorted_Z_vehicles = self.get_sorted_Z_vehicles(vehicles.vehicles_objs)
        sorted_Z_vehicles = sorted(vehicles.vehicles_objs, key=lambda x: x.distance, reverse=True)
        for vcl in sorted_Z_vehicles:
            if not (vcl.visibility == 'visible'):
                continue
            #vcl = vehicles.vehicles_objs[v]
            Z = vcl.distance # for now - need to change such that distance would be along
                             # the lane
            X = lane_models[vcl.lane_idx].Z2X(Z)

            dX_dZ = lane_models[vcl.lane_idx].dX_dZ(Z)
            vcl_yaw_angle = np.math.atan(dX_dZ)
            # self.draw_single_vehicle_on_top_view(tvi, vcl, X, Z)
            self.draw_single_vehicle(vcl, X, Z, dX_dZ)
            v_cntr += 1

    def get_pixel_center_ray_Z_equation_paramteres(self, x, y):
        dX_dZ = (x - self.x_center) / self.fl
        dY_dZ = -(y - self.y_center) / self.fl
        ''' Z = a * X + b * Y + c    // c===0 '''
        dY_dZ_inv = 100000
        dX_dZ_inv = 100000

        if abs(dX_dZ) > epsilon:
            dX_dZ_inv = 1 / dX_dZ
        if abs(dY_dZ) > epsilon:
            dY_dZ_inv = 1 / dY_dZ
        res = {'dXdZ': dX_dZ, 'dYdZ': dY_dZ, '1/dXdZ': dX_dZ_inv, '1/dYdZ': dY_dZ_inv}

        return res

    def set_pixel(self, i, j, val):
        """setting a val into pixel"""
        if (i > 0) and (i < self.img.shape[0]) and (j > 0) and (j < self.img.shape[1]):
            self.img[i, j] = val

    def display(self, save_path=None):
        plt.figure("tv_img")
        plt.imshow(self.img)
        plt.show()
        if save_path is not None:
            cv2.imwrite(save_path, self.img)

    def save(self, save_path, FV_point_center_host_in_100m, points_list_save_path):

        # np.save(points_list_save_path, self.drawed_points)
        self.draw_horizon_cross(FV_point_center_host_in_100m)
        # self.draw_exit()
        # self.draw_merge()
        #self.dict_to_dump_vehicels()
        cv2.imwrite(save_path, self.img)

    def vcls_on_fvi_list(self):

        # self.vcl_centers_x = list()
        # self.vcl_centers_y = list()
        # self.vcl_widths = list()
        # self.vcl_heights = list()

        # for vcl_idx in range(len(self.vcl_centers_x)):
        #     self.vcl_centers_x.append(self.vcl_centers_x)
        #     self.vcl_centers_y.append(self.vcl_centers_y)
        #     self.vcl_widths.append(self.vcl_widths)
        #     self.vcl_heights.append(self.vcl_heights)
        #
        # self.dict_to_dump = dict()
        # self.dict_to_dump['front_view_vehicles_center_x'] = self.vcl_centers_x
        # self.dict_to_dump['front_view_vehicles_center_y'] = self.vcl_centers_y
        # self.dict_to_dump['front_view_vehicles_width'] = self.vcl_widths
        # self.dict_to_dump['front_view_vehicles_height'] = self.vcl_heights
        return [self.vcl_centers_x, self.vcl_centers_bottom_y, self.vcl_widths, self.vcl_heights]


    def save_seg_images(self, seg_img_save_path, cropped_img_save_path,
                        seg_points_list_save_path, cropped_points_list_save_path):

        seg_image_full = np.zeros(shape=(self.img.shape[0], self.img.shape[1]), dtype=np.uint8)

        solid_indices = np.where(self.img[:, :, 0] == type_name2color('solid')[0])
        dashed_indices = np.where(self.img[:, :, 0] == type_name2color('dashed')[0])
        vcl_indices = np.where(self.img[:, :, 0] == type_name2color('vehicle')[0])
        seg_image_full[solid_indices[0], solid_indices[1]] = type_name2type_idx('solid')
        seg_image_full[dashed_indices[0], dashed_indices[1]] = type_name2type_idx('dashed')
        seg_image_full[vcl_indices[0], vcl_indices[1]] = type_name2type_idx('vehicle')

        seg_image = seg_image_full[CROP_FULL_IMG[1]:CROP_FULL_IMG[1] + CROP_FULL_IMG[3],
                                   CROP_FULL_IMG[0]:CROP_FULL_IMG[0] + CROP_FULL_IMG[2]]

        seg_image = cv2.resize(seg_image, SEG_IMG_DIMS)
        cv2.imwrite(seg_img_save_path, seg_image)
        seg_cropped_img = seg_image_full[CROP_CROPPED_IMG[1]:CROP_CROPPED_IMG[1] + CROP_CROPPED_IMG[3],
                                        CROP_CROPPED_IMG[0]:CROP_CROPPED_IMG[0] + CROP_CROPPED_IMG[2]]
        cv2.imwrite(cropped_img_save_path, seg_cropped_img)
#     def dict_to_dump_vehicels(self):

        # seg_drawed_points = self.translate_points_to_resized(self.drawed_points)
        # seg_cropped_drawed_points = self.translate_points_to_cropped(self.drawed_points)
        # np.save(seg_points_list_save_path, seg_drawed_points)
        # np.save(cropped_points_list_save_path, seg_cropped_drawed_points)


if __name__ == "__main__":

    for i in range(1000):
        tv = TV_image()
        tv.draw_lanes()
        # tv.display_cropped()

        length_pnts_ = 2000
        width_pnts_ = 400
        length_m_ = 200
        width_m_ = 40
        height_m_ = 10

        rm_pc = RoadManifoldPointCloud(length_pnts_, width_pnts_, length_m_, width_m_)
        lrm = LinearRoadModel(0,0)
        dir_path = 'D:\phantomAI\data\synthesized_data'
        filename_ = 'road_manifold_point_cloud_2019_10_16__14_11_17.npz'
        # filename_ = 'road_manifold_point_cloud_2019_10_17__19_42_15.npz'
        full_path = os.path.join(dir_path, filename_)

        rm_pc.load_point_cloud_from_file(full_path)

        #plt.imshow(rm_pc.point_cloud[:, :, 1])
        # fvi_factory = FVI_Factory(width_pix=1208, height_pix=1920, focal_length=2000, camera_height=1.2, x_center=959, y_center=603)
        # fvi = fvi_factory.draw_from_TV_image_and_linear_road_model(tv, lrm)
        # fvi = fvi_factory.draw_from_TV_image_and_manifold(tv,rm_pc)
        dt_string = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        fvi_filename = "front_view_image_" + dt_string + ".png"
        tvi_filename = "top_view_image_" + dt_string + ".png"
        # cv2.imwrite(os.path.join(dir_path, fvi_filename), fvi.img)
        cv2.imwrite(os.path.join(dir_path, tvi_filename), tv.img)

        # print("fvi.shape", fvi.img.shape)
        # fvi.display()
