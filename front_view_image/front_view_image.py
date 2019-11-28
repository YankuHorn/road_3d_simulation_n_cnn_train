import matplotlib.pyplot as plt
import os
from road_manifold.road_manifold import RoadManifoldPointCloud
from road_manifold.linear_road_model import LinearRoadModel
from road_top_view_image.tv_image import TV_image
import cv2
from datetime import datetime
import time
from tools.random_tools import *
from tools.draw_tools import draw_rect, type2color
dX_threshold = 0.3
dY_threshold = 0.3

epsilon = 0.000000001


class FVI_Factory:
    def __init__(self, width_pix, height_pix, focal_length, camera_height, x_center, y_center):
        self.fvi_img_w = height_pix
        self.fvi_img_h = width_pix
        self.fvi_fl = focal_length
        self.fvi_ch = camera_height

        self.fvi_x_center = x_center
        self.fvi_y_center = y_center

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

    def find_line_point_intersection_linear(self, pixel_ray_equation_parameters, camH, lrm_pitch_rad, lrm_roll_rad):
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
        a = (pixel_ray_equation_parameters['dYdZ'] - np.math.tan(lrm_pitch_rad) - np.math.tan(lrm_roll_rad) *
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
        FVI = FV_image(self.fvi_img_w, self.fvi_img_h, self.fvi_fl, self.fvi_ch, self.fvi_x_center, self.fvi_y_center)
        # for i in range(self.fvi_y_center + 10, self.fvi_y_center + 15): #
        for i in range(self.fvi_img_h):
            if i==705 :
                print("going over image i=",i, "out of ",self.fvi_img_h)
            # if i==600:
            #     print("i=650")
            # for j in range(self.fvi_x_center-1, self.fvi_x_center+1): # range(self.fvi_img_w):
            for j in range(self.fvi_img_w):
                if j%50==0:
                    print('for debug')
                pixel_ray_equation_parameters = FVI.get_pixel_center_ray_Z_equation_paramteres(j, i)
                X, Y, Z = self.find_line_point_intersection_linear(pixel_ray_equation_parameters, self.fvi_ch,
                                                                   linear_road_model.pitch_rad, linear_road_model.roll_rad)
                if Z is not None:
                    # if (Z < 500 and (X%10 == 0)):
                        # print("bp")
                    x, y = tv_image.XZ2mn(X, Z)
                    if x is not None and y is not None:
                        FVI.set_pixel(i, j, tv_image.img[y, x])
        return FVI


class FV_image:
    def __init__(self, width_pix, height_pix, focal_length, camera_height, x_center, y_center):
        self.img = np.zeros((height_pix, width_pix, 3), dtype=np.int16)
        self.img_w = width_pix
        self.img_h = height_pix
        self.camH = camera_height
        self.fl = focal_length

        self.x_center = x_center
        self.y_center = y_center

    def XZ2xy(self, X, Z, height=None):
        """
        (y - y_center) / focal_length = height / Z
        (x - x_center) / focal_length = X / Z
         = >
         y = (height / Z) * focal_length + y_center
         y = (X / Z) * focal_length + x_center
        """
        if height is None:
            height = self.camH
        y = int(round((height / Z) * self.fl + self.y_center))
        x = int(round((X / Z) * self.fl + self.x_center))
        if (x > 0) and (y > 0) and (x < self.img_w) and (y < self.img_h):
            return x, y
        else:
            return None, None

    def draw_single_vehicle(self, vcl, X, Z, vcl_yaw_angle):
        visible_width = vcl.size['w'] * np.math.cos(vcl_yaw_angle)

        X_l = X - 0.5 * visible_width
        X_r = X + 0.5 * visible_width
        Z = Z
        vcl_height = vcl.size['h']
        top_of_vehicle_height = self.camH - vcl_height
        # l = left ; r = right ; b = bottom ; t = top
        x_lb, y_lb = self.XZ2xy(X_l, Z)
        x_rb, y_rb = self.XZ2xy(X_r, Z)
        x_lt, y_lt = self.XZ2xy(X_l, Z, height=top_of_vehicle_height)
        x_rt, y_rt = self.XZ2xy(X_r, Z, height=top_of_vehicle_height)

        color = type2color('vehicle')
        if (x_lb is not None) and (x_rb is not None) and (x_lt is not None) and (x_rt is not None):
            self.img = draw_rect(self.img, y_lb, x_lb, y_rt, x_rt, color, fill_clr=[0, 0, 0], width=5)

    def draw_vehicles(self, vehicles, lane_models):
        v_cntr = 0
        # sorted_Z_vehicles = self.get_sorted_Z_vehicles(vehicles.vehicles_objs)
        sorted_Z_vehicles = sorted(vehicles.vehicles_objs, key=lambda x: x.distance, reverse=True)
        for vcl in sorted_Z_vehicles:
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
        if save_path is not None:
            cv2.imwrite(save_path, self.img)


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
