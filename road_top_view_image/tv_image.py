
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tools.draw_tools import draw_rect, type_name2color

# ========================================
# Written by Kobi horn, Oct 2019
# ========================================

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

class drawed_point:
    def __init__(self, r, c, clr, point_type, is_exit=False, is_merge=False):
        self.r = r
        self.c = c
        self.clr = clr
        self.point_type = point_type,
        self.is_exit = is_exit
        self.is_merge = is_merge

class TV_image:
    def __init__(self, width_pix=4800, height_pix=24000, width_m=40, height_m=200):
        self.img = np.zeros((height_pix, width_pix, 3), dtype=np.int16)
        self.img_w = width_pix
        self.img_h = height_pix
        self.pixel_width = width_m/width_pix
        self.pixel_height = height_m / height_pix
        self.dimensions = (self.img_w, self.img_h)
        self.width_m = width_m
        self.height_m = height_m
        self.exit_merge_points = np.zeros((height_pix, width_pix, 2), dtype=bool)

    def XZ2mn(self, X, Z):
        m = int((self.width_m/2 + X ) / self.pixel_width)
        n = int(Z / self.pixel_height)
        if (m > 0 ) and (m < self.img.shape[1]) and (n > 0) and (n < self.img.shape[0]):
            return m, n
        else:
            return None, None

    def Z2n(self, Z):
        n = int(Z / self.pixel_height)
        if (n > 0) and (n < self.img.shape[0]):
            return n
        else:
            return None

    def n2Z(self, n):
        Z = n * self.pixel_height
        return Z

    def draw_lines(self, lines):
        for line in lines:
            self.draw_line(line)

    def draw_vehicles(self, vehicles, lane_models):
        v_cntr = 0

        for l, lane_model in enumerate(lane_models):
            num_vcls_in_lane = vehicles.num_vehicles_per_lane[l]
            for v in range(v_cntr, v_cntr + num_vcls_in_lane):
                vcl = vehicles.vehicles_objs[v]
                Z = vcl.distance # for now - need to change such that distance would be along
                                 # the lane
                X = lane_model.Z2X(Z)
                # self.draw_single_vehicle_on_top_view(tvi, vcl, X, Z)
                self.draw_single_vehicle(vcl, X, Z)
                v_cntr += 1

    def draw_single_vehicle(self, vcl, X, Z):
        X_l = X - 0.5 * vcl.size['w']
        X_r = X + 0.5 * vcl.size['w']
        Z_c = Z
        Z_f = Z + vcl.size['l']
        m_lc, n_lc = self.XZ2mn(X_l, Z_c)
        m_lf, n_lf = self.XZ2mn(X_l, Z_f)
        m_rc, n_rc = self.XZ2mn(X_r, Z_c)
        m_rf, n_rf = self.XZ2mn(X_r, Z_f)
        color = type_name2color('vehicle')
        if (m_lc is not None) and (m_rc is not None) and (m_lf is not None) and (m_rf is not None):
            self.img = draw_rect(self.img, n_lc, m_lc, n_rf, m_rf, color)

    def draw_line(self, line):

        for i in range(line.num_segments):
            for Z in frange(line.Z_ranges[i][0], line.Z_ranges[i][1], self.pixel_height):
                if (Z > line.gap['begin']) and (Z < line.gap['begin'] + line.gap['length']):
                    continue
                X_center = line.lane_models[i].Z2X(Z)
                half_width = line.widths[i] * 0.5
                for X in frange(X_center - half_width, X_center + half_width, self.pixel_width):
                    m, n = self.XZ2mn(X, Z)
                    color = type_name2color(line.solidashed_types[i])
                    if line.is_exit[i]:
                        color = type_name2color('solid_exit')
                    if line.is_merge[i]:
                        color = type_name2color('solid_merge')
                    if m is not None:
                        self.img[n, m] = np.asarray(color)
                        self.exit_merge_points[n, m] = [line.is_exit[i], line.is_merge[i]]
    # def draw_lanes(self, main_lane_model, exit_lane_model=None):
    #     #    def draw_lanes(self,
    #
    #     # for i in range(self.img_h):
    #     #     if i % height < 100:
    #     #         val = 254
    #     #         self.img[i, :, :] = [val, val, val]
    #     # num_lines = num_lanes + 1
    #     # main_a0 = 0
    #     # main_a1 = -0.1
    #     # main_a2 = 0.0001
    #     # main_a3 = 0.00001
    #     # main_a4 = 0
    #     # main_lane_model = LaneModel()
    #     # main_lane_model.load_model(main_a0, main_a1, main_a2, main_a3, main_a4)
    #     color_dashed = [255, 255, 0]
    #     color_solid = [255, 0, 255]
    #     lane_mark_width = 0.2
    #
    #     self.draw_line(main_lane_model, delta_X=-5.4, width=lane_mark_width, color=color_solid, Z_range=[0, 200.0])
    #     self.draw_line(main_lane_model, delta_X=-1.8, width=lane_mark_width, color=color_dashed, Z_range=[0, 200.0])
    #     if exit_lane_model is None:
    #         self.draw_line(main_lane_model, delta_X=1.8, width=lane_mark_width, color=color_solid, Z_range=[0, 200.0])
    #     else:
    #         self.draw_line(main_lane_model, delta_X=1.8, width=lane_mark_width, color=color_solid, Z_range=[30, 200.0])
    #         #Vexit_lane_model = LaneModel()
    #         # exit_lane_model.load_model(main_a0, main_a1 + 0.2, 0.001, 0.00001, 0)
    #         self.draw_line(exit_lane_model, delta_X=1.8, width=lane_mark_width, color=color_solid, Z_range=[0, 80.0])
    #         self.draw_line(exit_lane_model, delta_X=-1.8, width=lane_mark_width, color=color_solid, Z_range=[30., 80.0])

    def display(self, save_path=None):
        plt.figure("tv_img")
        plt.imshow(self.img)
        plt.show()

    def save(self, save_path=None):
        if save_path is not None:
            cv2.imwrite(save_path, self.img)

    def display_cropped(self, save_path=None):
        plt.figure("tv_img_crop")
        plt.imshow(self.img[:1000,250:-250,:])
        plt.show()
        print("PIPIP AL HARATZIF")
        if save_path is not None:
            cv2.imwrite(save_path, self.img)
if __name__ == "__main__":
    tv = TV_image()
    # tv.draw_lanes()

    tv.display()