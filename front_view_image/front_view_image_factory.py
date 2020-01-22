import time
from tools.draw_tools import color2type_idx
import numpy as np
from tools.random_tools import get_rand_range
from front_view_image.front_view_image import FV_image

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
                        if not (np.all(tv_image.img[y, x] == [0, 0, 0])):
                            FVI.drawed_points.append([i, j, color2type_idx(tv_image.img[y, x])])
        return FVI
