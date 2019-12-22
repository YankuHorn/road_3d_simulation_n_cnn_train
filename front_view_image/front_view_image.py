import matplotlib.pyplot as plt
import os
from road_manifold.road_manifold import RoadManifoldPointCloud
from road_manifold.linear_road_model import LinearRoadModel
from road_top_view_image.tv_image import TV_image
import cv2
from datetime import datetime
import time
import csv
from tools.draw_tools import draw_rect, type2color, draw_line2, draw_horizon_cross
import numpy as np
from tools.random_tools import get_rand_range
dX_threshold = 0.3
dY_threshold = 0.3

epsilon = 0.000000001

SEG_IMG_DIMS = (512, 288)
CROP_FULL_IMG = (0, 120, 1920, 1080)
CROP_CROPPED_IMG = (704, 516, 512, 288)

class FVI_Factory:
    def __init__(self, width_pix, height_pix, focal_length, camera_height, x_center, y_center):
        self.fvi_img_w = height_pix
        self.fvi_img_h = width_pix
        self.fvi_fl = focal_length
        self.fvi_ch = camera_height

        self.fvi_x_center = x_center
        self.fvi_y_center = y_center

        # self.cam_pitch_deg = get_rand_range(-3, 3)
        # self.cam_yaw_deg = get_rand_range(-5, 5)
        self.cam_roll_deg = get_rand_range(-2, 2)

        # self.cam_pitch_rad = self.cam_pitch_deg * np.pi / 180
        # self.cam_yaw_rad = self.cam_yaw_deg * np.pi / 180
        self.cam_roll_rad = self.cam_roll_deg * np.pi / 180

    @staticmethod
    def get_dist_to_point(equation_params, X, Y, Z):
        """ todo """
        return 0

    @staticmethod
    def get_dX_in_XY_plane(equation_params, point_cloud):
        """ calculates dX of a line to a point on XY plane set by Z """
        Z = point_cloud[:, :, 2]
        X = point_cloud[:, :, 0]

        X_line_at_plane = equation_params['dXdZ'] * Z
        dX = X_line_at_plane - X

        return dX

    def get_dY_in_XY_plane(self, equation_params, point_cloud):
        """ calculates dY of a line to a point on XY plane set by Z """
        Z = point_cloud[:, :, 2]
        Y = point_cloud[:, :, 1] - self.fvi_ch
        Y_line_at_plane = equation_params['dYdZ'] * Z

        dY = Y_line_at_plane - Y
        return dY

    def find_line_point_intersection(self, equation_params, point_cloud):
        init_time = time.time()

        dX = self.get_dX_in_XY_plane(equation_params, point_cloud)
        a_time = time.time()
        dx_time = a_time - init_time
        dY = self.get_dY_in_XY_plane(equation_params, point_cloud)
        b_time = time.time()
        dy_time = b_time - a_time
        dX2 = dX * dX
        c_time = time.time()
        dx2_time = c_time - b_time
        dY2 = dY * dY
        d_time = time.time()
        dy2_time = d_time - c_time
        dist_on_XY_plane_2 = dX2 + dY2
        min_inds = np.unravel_index(np.argmin(dist_on_XY_plane_2, axis=None), dist_on_XY_plane_2.shape)
        e_time = time.time()
        min_inds_time = e_time - d_time
        #print("point_cloud_shape", point_cloud.shape, "dx", dx_time, "dy", dy_time, "dx2", dx2_time, "dy2", dy2_time, "min inds", min_inds_time)
        return min_inds
    #def fvi_xy_2_XZ_with_linear_road_model(self, fl, ch, x, y, rmm_pitch, rmm_yaw, rmm_tilt):

    def find_line_point_intersection_linear(self, pixel_ray_equation_parameters, camH):
        """
        plane_equation:
        (1) Y = tan(pitch) * Z + tan(roll) * X - camH
        pixel ray equation:
        (2) X = dX/dZ * Z
        (3) Y = dY/dZ * Z
        hence (by putting eq. (2) and eq. (3) into eq (1)):
        dY_dZ *  Z = tan(pitch) * Z + tan(roll) * dX_dY * Z - camH
        ( dy_dZ - tan(pitch) -  dX_dY * tan(roll) ) * Z = -camH
        (4) Z = -camH / (dY_dZ - tan(pitch) - tan(roll) * dX_dY)
        and Once we found Z, X and Y are easy to find with eq. (2) and (3)
        """

        total_roll = self.cam_roll_rad

        # yaw is handled in the lane building # total_yaw = 0 + self.cam_yaw_deg
        a = (pixel_ray_equation_parameters['dYdZ'] - np.math.tan(total_roll) *
         pixel_ray_equation_parameters['dXdZ'])
        if a < 0:
            Z = -camH / a
        else:
            return None, None, None
        X = Z * pixel_ray_equation_parameters['dXdZ']
        Y = Z * pixel_ray_equation_parameters['dYdZ']
        return X, Y, Z

    def draw_from_TV_image_and_manifold(self, tv_image, road_manifold_point_cloud):
        FVI = FV_image(self.fvi_img_w, self.fvi_img_h, self.fvi_fl, self.fvi_ch, self.fvi_x_center, self.fvi_y_center)
        for i in range(self.fvi_img_h):
            if i%50 ==0 :
                print("going over image i=",i, "out of ",self.fvi_img_h)
            for j in range(self.fvi_img_w):
                if j == self.fvi_x_center:
                    if i > self.fvi_y_center:

                        print("bla bla")
                pixel_ray_equation_parameters = FVI.get_pixel_center_ray_Z_equation_paramteres(j, i)
                m, n = self.find_line_point_intersection(pixel_ray_equation_parameters, road_manifold_point_cloud.point_cloud)
                FVI.set_pixel(i, j, tv_image.img[m, n])
        return FVI

    def draw_from_TV_image_and_linear_road_model(self, tv_image, linear_road_model):
        FVI = FV_image(self.fvi_img_w, self.fvi_img_h, self.fvi_fl, self.fvi_ch, self.fvi_x_center, self.fvi_y_center,
                       self.cam_roll_rad)
        # for i in range(self.fvi_y_center + 10, self.fvi_y_center + 15): #
        for i in range(self.fvi_img_h):
#             if i==705 :
#                 print("going over image i=",i, "out of ",self.fvi_img_h)
            # if i==600:
            #     print("i=650")
            # for j in range(self.fvi_x_center-1, self.fvi_x_center+1): # range(self.fvi_img_w):
            for j in range(self.fvi_img_w):
                # if j%50==0:
                #     print('for debug')
                pixel_ray_equation_parameters = FVI.get_pixel_center_ray_Z_equation_paramteres(j, i)

                X, Y, Z = self.find_line_point_intersection_linear(pixel_ray_equation_parameters, self.fvi_ch)
                if Z is not None:
                    # if (Z < 500 and (X%10 == 0)):
                        # print("bp")
                    x, y = tv_image.XZ2mn(X, Z)
                    if x is not None and y is not None:
                        [exit_p, merge_p] = tv_image.exit_merge_points[y, x]
                        if exit_p:
                            FVI.exit_points.append([i, j])
                        if merge_p:
                            FVI.merge_points.append([i, j])

                        FVI.set_pixel(i, j, tv_image.img[y, x])
        return FVI


class FV_image:
    def __init__(self, width_pix, height_pix, focal_length, camera_height, x_center, y_center, cam_roll=0):
        self.img = np.zeros((height_pix, width_pix, 3), dtype=np.int16)
        self.img_w = width_pix
        self.img_h = height_pix
        self.camH = camera_height
        self.fl = focal_length

        self.x_center = x_center
        self.y_center = y_center

        self.vehicles = list()

        self.vcls_lines = list()
        self.cam_roll = cam_roll

        self.exit_points = list()
        self.merge_points = list()
        self.exit_decision = "no_exit"
        self.merge_decision = "no_merge"

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

        color = type2color('vehicle')
        if (x_lb is not None) and (x_rb is not None) and (x_lt is not None) and (x_rt is not None):
            self.img = draw_rect(self.img, y_lb, x_lb, y_rt, x_rt, color, fill_clr=[0, 0, 0], width=5)
            self.remove_exit_points(x_lb, x_rb, y_lt, y_lb)
            self.remove_merge_points(x_lb, x_rb, y_lt, y_lb)

        # ['#', 'ID', 'class_name', 'box_valid', 'x', 'y', 'width', 'height', 'rear_valid', 'x', 'y', 'width', 'height']
        vcl_line = [vcl.ID, 'vehicle', 0, 0 , 0 ,0, 0, 1, x_lt, y_lt, visible_width, vcl_height]
        self.vcls_lines.append(vcl_line)

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

    def draw_horizon_cross(self):
        draw_horizon_cross(self.img, self.y_center, self.x_center, cam_roll=self.cam_roll, clr=[0, 0, 255])
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

    def save(self, save_path=None):

        self.draw_horizon_cross()
        self.draw_exit()
        self.draw_merge()

        if save_path is not None:
            cv2.imwrite(save_path, self.img)

    def save_seg_images(self, seg_img_save_path, cropped_img_save_path):
                        #,csv_objs_save_path, csv_crop_objs_save_path):

        seg_image_full = np.zeros(shape=(self.img.shape[0], self.img.shape[1]), dtype=np.uint8)

        solid_indices = np.where(self.img[:, :, 0] == type2color('solid')[0])
        dashed_indices = np.where(self.img[:, :, 0] == type2color('dashed')[0])
        vcl_indices = np.where(self.img[:, :, 0] == type2color('vehicle')[0])
        seg_image_full[solid_indices[0], solid_indices[1]] = 3
        seg_image_full[dashed_indices[0], dashed_indices[1]] = 4
        seg_image_full[vcl_indices[0], vcl_indices[1]] = 8

        seg_image = seg_image_full[CROP_FULL_IMG[1]:CROP_FULL_IMG[1] + CROP_FULL_IMG[3],
                                   CROP_FULL_IMG[0]:CROP_FULL_IMG[0] + CROP_FULL_IMG[2]]

        seg_image = cv2.resize(seg_image, SEG_IMG_DIMS)
        cv2.imwrite(seg_img_save_path, seg_image)
        seg_cropped_img = seg_image_full[CROP_CROPPED_IMG[1]:CROP_CROPPED_IMG[1] + CROP_CROPPED_IMG[3],
                                        CROP_CROPPED_IMG[0]:CROP_CROPPED_IMG[0] + CROP_CROPPED_IMG[2]]
        cv2.imwrite(cropped_img_save_path, seg_cropped_img)
        # ID, class_name, box_valid, x, y, width, height, rear_valid, x, y, width, height
        # 19, Vehicle, 1, 305.00, 133.00, 30.00, 22.00, 1, 307.00, 133.00, 28.00, 23.00
        # header = ['#', 'ID', 'class_name', 'box_valid', 'x', 'y', 'width', 'height', 'rear_valid', 'x', 'y', 'width', 'height']

        # with open(csv_objs_save_path, "w", newline='') as f:
        #     writer = csv.writer(f, delimiter=',')
        #     writer.writerow(header)  # write the header
        #     # write the actual content line by line
        #     for line in self.vcls_lines:
        #         [ID, type, _, _, _, _, _, validness, x_lt, y_lt, visible_width, vcl_height] = line
        #         x_lt_after_crop = x_lt - CROP_FULL_IMG[0]
        #         y_lt_after_crop = y_lt - CROP_FULL_IMG[1]
        #         line_after_crop = [ID, type, 0, 0, 0, 0, 0, validness, x_lt_after_crop, y_lt_after_crop, visible_width, vcl_height]
        #         writer.writerow(line_after_crop)
        # with open(csv_crop_objs_save_path, "w", newline='') as f:
        #     writer = csv.writer(f, delimiter=',')
        #     writer.writerow(header)  # write the header
        #     # write the actual content line by line
        #     for line in self.vcls_lines:
        #         [ID, type, _, _, _, _, _, validness, x_lt, y_lt, visible_width, vcl_height] = line
        #         x_lt_after_crop = x_lt - CROP_FULL_IMG[0]
        #         y_lt_after_crop = y_lt - CROP_FULL_IMG[1]
        #         line_after_crop = [ID, type, 0, 0, 0, 0, 0, validness, x_lt_after_crop, y_lt_after_crop, visible_width, vcl_height]
        #         writer.writerow(line_after_crop)


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
        fvi_factory = FVI_Factory(width_pix=1208, height_pix=1920, focal_length=2000, camera_height=1.2, x_center=959, y_center=603)
        fvi = fvi_factory.draw_from_TV_image_and_linear_road_model(tv, lrm)
        # fvi = fvi_factory.draw_from_TV_image_and_manifold(tv,rm_pc)
        dt_string = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        fvi_filename = "front_view_image_" + dt_string + ".png"
        tvi_filename = "top_view_image_" + dt_string + ".png"
        cv2.imwrite(os.path.join(dir_path, fvi_filename), fvi.img)
        cv2.imwrite(os.path.join(dir_path, tvi_filename), tv.img)

        print("fvi.shape", fvi.img.shape)
        # fvi.display()
