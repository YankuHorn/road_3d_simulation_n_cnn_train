import numpy as np
import random
# import csv
# import pandas
import matplotlib.pyplot as plt
import os
from datetime import datetime

class Gaussian:
    def __init__(self, mean, std, angle, magnitude):
        self.mean = mean
        self.std = std
        self.angle = angle
        self.magnitude = magnitude

    def get_val(self, x):
        # todo - add the angle
        sigma = np.asarray([[self.std[0] * self.std[0], 0], [0, self.std[1] * self.std[1]]])
        sqrt_det_sigma = self.std[0] * self.std[1]
        sigma_inv = np.linalg.inv(sigma)
        exp_arg = (-0.5) * np.dot(np.dot((x-self.mean).T, sigma_inv), (x-self.mean))
        return self.magnitude * (1/(2 * np.pi * sqrt_det_sigma)) * np.exp(exp_arg)

    def get_mean(self):
        return self.mean

    def get_std(self):
        return self.std

    def get_angle(self):
        return self.angle


class RoadManifoldPointCloud:
    """
    Special, thin point cloud - similar to top-view height image - X (lateral) and Z (longitudinal) axes are
    splited into "pixels" in each the value is the Y (height)
    """
    def __init__(self, length_pnts, width_pnts, length_m, width_m):
        self.point_cloud = np.zeros((length_pnts, width_pnts, 3), dtype=float)
        self.width_scale = width_m / width_pnts
        self.length_scale = length_m / length_pnts
        self.width_scale_inv =  width_pnts / width_m
        self.length_scale_inv = length_pnts / length_m

    def load_point_cloud_from_file(self, filename):
        data = np.load(filename)
        self.point_cloud = data['arr_0']

    def set_val(self, i, j, X, Y, Z):
        self.point_cloud[i, j, :] = np.asarray([X, Y, Z])

    def j2X(self, j):
        return self.width_scale * j

    def i2Z(self, i):
        return self.length_scale * i

    def X2j(self, X):
        return self.width_scale_inv * X

    def Z2i(self, Z):
        return self.length_scale_inv * Z

    def XZ2Y(self, X, Z):
        return self.point_cloud[self.Z2i(Z), self.X2j(X)]

    def ij2Y(self, i, j):
        return self.point_cloud[i, j]


class RoadManifoldsFactory:
    def __init__(self, range_mean, range_std, range_num_gaussians, range_magnitude, range_angle):
        """
        :param num_gaus:
        :param range_mean:
        :param range_std:
        :param range_num_gaussians:
        """
        self.range_mean = range_mean
        self.range_std = range_std
        self.range_num_gaussians = range_num_gaussians
        self.range_magnitude = range_magnitude
        self.range_angle = range_angle

    def get_manifold(self):
        num_gaussians = random.randint(self.range_num_gaussians[0], self.range_num_gaussians[1])
        gaussians = list()
        for gaus_idx in range(num_gaussians):

            mean = self.range_mean[0] + np.random.rand(2) * self.range_mean[1]
            std = self.range_std[0] + np.random.rand(2) * self.range_std[1]
            angle = self.range_angle[0] + np.random.rand() * self.range_angle[1]
            magnitude = self.range_magnitude[0] + np.random.rand() * self.range_magnitude[1]

            rand_gaussian = Gaussian(mean, std, angle, magnitude)
            gaussians.append(rand_gaussian)
        res = RoadManifold()
        res.set_gaussians(gaussians)
        return res


class RoadManifold:

    def __init__(self):
        self.gaussians = None
        self.rm_pc = None

    def set_gaussians(self, gaussians):
        self.gaussians = gaussians

    def create_point_cloud(self, length_pnts=200, width_pnts=40,
                           length_m=200, width_m=40, height_m=10):

        self.rm_pc = RoadManifoldPointCloud(length_pnts, width_pnts, length_m, width_m)

        for i in range(length_pnts):
            print('i ', i, 'out of ', length_pnts)
            for j in range(width_pnts):
                X = self.rm_pc.j2X(j)
                Z = self.rm_pc.i2Z(i)
                Y = self.init_Y(X, Z)
                self.rm_pc.set_val(i, j, X, Y, Z)

    def log_point_cloud(self, filename):
        np.savez(filename, self.rm_pc.point_cloud)

    def init_Y(self, X, Z):
        Y = 0
        for gaussian in self.gaussians:
            Y += gaussian.get_val(np.asarray([X, Z]))
        return Y

    def XZ2Y(self, X, Z):
        return self.rm_pc.XZ2Y(X, Z)

    def ij2Y(self, i, j):
        return self.rm_pc.ij2Y(i, j)

    def get_pixel_center_ray(self, pixel_center_ray):
        return 0


if __name__ == "__main__":
    range_mean = [[10, 50], [10, 50]]  # -150 150 -150
    range_std = [[50, 50], [50, 50]]  # 25 25 250 250
    range_num_gaussians = [0, 0]  # 1 7
    range_magnitude = [10, 10]  # 0 50
    range_angle = [0, 0]
    new_rm = False
    if new_rm:
        rmf = RoadManifoldsFactory(range_mean, range_std, range_num_gaussians, range_magnitude, range_angle)
        road_manifold = rmf.get_manifold()
        road_manifold.create_point_cloud()
        dt_string = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        dir_path = 'D:\phantomAI\data\synthesized_data'
        filename_ = 'road_manifold_point_cloud_' + dt_string
        full_path = os.path.join(dir_path, filename_)
        road_manifold.log_point_cloud(full_path)
    else:
        length_pnts_ = 2000
        width_pnts_ = 400
        length_m_ = 200
        width_m_ = 40
        height_m_ = 10

        rm_pc = RoadManifoldPointCloud(length_pnts_, width_pnts_, length_m_, width_m_, height_m_)
        dir_path = 'D:\phantomAI\data\synthesized_data'
        filename_ = 'road_manifold_point_cloud_2019_10_16__14_11_17.npz'
        full_path = os.path.join(dir_path, filename_)

        rm_pc.load_point_cloud_from_file(full_path)

    plt.imshow(road_manifold.rm_pc.point_cloud[:, :, 1])
    plt.show()
    print("for the breakpoint")