import numpy as np
import random as random
from road_top_view_image.tv_image import TV_image
from road_manifold.linear_road_model import LinearRoadModel
from front_view_image.front_view_image import FVI_Factory
import copy
import os
from datetime import datetime
import json
from tools.random_tools import get_rand_out_of_list_item, get_rand_range, get_rand_list_item
from road_topology.lane_model import LaneModel


class Line:
    def __init__(self, lane_models, Z_ranges, solidashed_types, widths):
        self.num_segments = len(lane_models)
        self.lane_models = lane_models
        self.Z_ranges = Z_ranges
        self.solidashed_types = solidashed_types
        self.widths = widths


def calc_Z_intersection_1_becomes_larget_than_2(model1, model2, min_val):

    enable_intersection_begin = False
    res = None
    for i in range(int(min_val), 200):
        X1 = model1.Z2X(i)
        X2 = model2.Z2X(i)
        if (X1 < X2) and (not enable_intersection_begin):
            enable_intersection_begin = True
        if enable_intersection_begin and (X1 > X2):
            res = i
            break
    if res is None:
        print("model1", model1.all(), "model2", model2.all())
    return res


class Lanes_factory:
    def __init__(self):
        #def get_rand_out_of_list_item(objects_list, num_items=None, objects_weights=None):

        self.num_main_lanes = get_rand_out_of_list_item([1, 2, 3, 4, 5], objects_weights=[0.1, 0.35, 0.35, 0.1, 0.1])
        self.num_lines = self.num_main_lanes + 1

        self.lanes_widths = get_rand_range(3.2, 4.0, num_items=self.num_main_lanes)

        self.lines_widths = get_rand_range(0.1, 0.3, num_items=self.num_lines)
        # z_range_beg_host = [0, 20]
        # self.lines_mark_z_ranges[0] = np.stack((get_rand_range(0, 20), get_rand_range(100, 150))).T
        self.lines_mark_z_ranges = np.stack((get_rand_range(0, 10, num_items=self.num_lines), get_rand_range(100, 150, num_items=self.num_lines))).T

        self.lane_model_Z_positions = np.asarray([-100, 0, 100, 150, 200])
        self.num_Z_points = len(self.lane_model_Z_positions)
        self.lane_model_X_deltas = get_rand_range(-10, 10, num_items=self.num_Z_points)
        self.lane_model_X_positions = np.zeros_like(self.lane_model_Z_positions, dtype=float)
        self.lane_model_X_positions[2:] = np.cumsum(self.lane_model_X_deltas[2:])
        self.lane_model_X_positions[0] = -self.lane_model_X_deltas[1]
        # self.lane_model_X_positions[0] = self.lane_model_X_positions[1] - self.lane_model_X_deltas[0]
        self.points = np.stack((self.lane_model_Z_positions, self.lane_model_X_positions)).T
        print("main lane points", self.points)
        self.lane_models = list()

        # self.is_exit_split_type = get_rand_list_item(["one2two_unmarked_V"])  #  ,   "one2one_dashed2solid2Y" ])
        self.is_exit_split_type = get_rand_list_item(["no_exit", "zero2one_unmarked_V", "one2one_dashed2solid2Y"])

        self.is_exit_beg_position = get_rand_range(-80, 50)  # -100, 100)
        self.is_exit_lane_width = get_rand_range(3.2, 4.0)
        self.is_exit_Z_positions = np.asarray([0, 100, 200, 300]) #, 200, 250, 300])
        #self.is_exit_Z_positions = np.asarray([0, 60])
        self.is_exit_num_Z_points = len(self.is_exit_Z_positions)
        num_points_around_Z_0 = 1
        self.is_exit_X_delta_deltas = get_rand_range(0, 10, num_items=self.is_exit_num_Z_points-num_points_around_Z_0)
        self.is_exit_X_deltas_from_main = np.zeros_like(self.is_exit_Z_positions, dtype=float)
        self.is_exit_X_deltas_from_main[num_points_around_Z_0:] = np.cumsum(self.is_exit_X_delta_deltas)
        self.exit_points_X = np.zeros_like(self.is_exit_X_deltas_from_main, dtype=float)
        self.exit_points_Z = np.zeros_like(self.is_exit_X_deltas_from_main, dtype=float)

        # self.is_exit_X_deltas_from_main[1] = 0
        # self.is_exit_X_deltas_from_main[0] = 0

        self.exit_lane_left_Z_range_end = get_rand_range(150, 180)
        self.exit_lane_right_Z_range_end = get_rand_range(150, 180)
        self.exit_lane_left_width = get_rand_range(0.1, 0.3)
        self.exit_lane_right_width = get_rand_range(0.1, 0.3)

        self.calc_center_lane_models()

        if self.is_exit_split_type == "no_exit":
            self.num_exit_lanes = 0
        elif self.is_exit_split_type == "zero2one_unmarked_V":
            self.num_exit_lanes = 1
            self.exit_model = self.calc_exit_leftmost_model()
            self.lane_models.append(self.exit_model)
        elif self.is_exit_split_type == "one2one_dashed2solid2Y":
            self.num_exit_lanes = 1
            self.exit_model = self.calc_exit_leftmost_model()
            self.lane_models.append(self.exit_model)

        self.num_total_lanes = len(self.lane_models)
        self.lines = self.build_lines()

    def get_lines(self):
        return self.lines

    def use_debug_set(self):
        self.num_main_lanes = 5
        self.num_lines = self.num_main_lanes + 1

        self.lanes_widths = 3.5 * np.ones(self.num_main_lanes)

        self.lines_widths = 0.15 * np.ones(self.num_lines)
        self.lines_mark_z_ranges = np.stack((np.zeros(self.num_lines), 200 * np.ones(self.num_lines))).T

        self.is_exit_split_type = "zero2one_unmarked_V"
        self.is_exit_beg_position = -10  # -100, 100)
        self.is_exit_lane_width = 3.5
        self.lane_model_Z_positions = np.asarray([-100, -50.0, 0, 50, 100, 150, 200])
        self.num_Z_points = len(self.lane_model_Z_positions)
        self.lane_model_X_deltas = [0,0,0,0,5,0,3]# * np.ones(self.num_Z_points)
        self.lane_model_X_positions = np.zeros_like(self.lane_model_Z_positions)
        self.lane_model_X_positions[3:] = np.cumsum(self.lane_model_X_deltas[3:])
        self.lane_model_X_positions[1] = -self.lane_model_X_deltas[1]
        self.lane_model_X_positions[0] = self.lane_model_X_positions[1] - self.lane_model_X_deltas[0]

        self.lane_model_X_positions = np.asarray([-3.3, 3.3, 0, 15.22, 16.4, 11, 8.5])
        self.points = np.stack((self.lane_model_Z_positions, self.lane_model_X_positions)).T
        self.exit_lane_left_Z_range_end = 200
        self.exit_lane_right_Z_range_end = 200
        self.lane_models = list()

    def dump_meta_data(self, top_view_image, front_view_image, filename):
        dict_to_dump = dict()

        dict_to_dump['num_lines'] = self.num_lines
        dict_to_dump['num_main_lanes'] = self.num_main_lanes

        dict_to_dump['lanes_widths'] = self.lanes_widths.tolist()

        dict_to_dump['lane_marks_widths'] = self.lines_widths.tolist()

        dict_to_dump['lines_mark_z_ranges'] = self.lines_mark_z_ranges.tolist()

        dict_to_dump['is_exit_split_type'] = self.is_exit_split_type
        dict_to_dump['is_exit_beg_position'] = self.is_exit_beg_position

        dict_to_dump['lane_model_Z_positions'] = self.lane_model_Z_positions.tolist()
        dict_to_dump['lane_model_X_positions'] = self.lane_model_X_positions.tolist()

        dict_to_dump['exit_lane_left_Z_range_end'] = self.exit_lane_left_Z_range_end
        dict_to_dump['exit_lane_right_Z_range_end'] = self.exit_lane_right_Z_range_end
        dict_to_dump['exit_lane_right_width'] = self.exit_lane_right_width
        dict_to_dump['exit_lane_left_width'] = self.exit_lane_left_width
        dict_to_dump['is_exit_lane_width'] = self.is_exit_lane_width
        dict_to_dump['is_exit_Z_positions'] = self.is_exit_Z_positions.tolist()
        dict_to_dump['exit_points_X'] = self.exit_points_X.tolist()
        dict_to_dump['exit_points_Z'] = self.exit_points_Z.tolist()
        if top_view_image is not None:
            dict_to_dump['top_view_img_total_width_meters'] = top_view_image.width_m
            dict_to_dump['top_view_img_total_height_meters'] = top_view_image.height_m
            dict_to_dump['top_view_img_pixel_width_meters'] = top_view_image.pixel_width
            dict_to_dump['top_view_img_pixel_height_meters'] = top_view_image.pixel_height
        if front_view_image is not None:
            dict_to_dump['front_view_image_camera_height'] = front_view_image.camH
            dict_to_dump['front_view_image_focal_length'] = front_view_image.fl
            dict_to_dump['front_view_image_x_center'] = front_view_image.x_center
            dict_to_dump['front_view_image_horizon'] = front_view_image.y_center

        with open(filename, 'w') as fp:
            json.dump(dict_to_dump, fp)

    def load_from_json(self, json_fp):

        with open(json_fp) as json_file:
            data = json.load(json_file)

        self.num_main_lanes = data['num_main_lanes']
        # self.num_lines = data['num_lines']
        self.num_lines = self.num_main_lanes + 1

        self.lanes_widths = np.asarray(data['lanes_widths'])

        self.lines_widths = np.asarray(data['lane_marks_widths'])
        self.lines_mark_z_ranges = np.asarray(data['lines_mark_z_ranges'])

        self.is_exit_split_type = data['is_exit_split_type']
        self.is_exit_beg_position = data['is_exit_beg_position']

        self.lane_model_Z_positions = np.asarray(data['lane_model_Z_positions'])
        self.lane_model_X_positions = np.asarray(data['lane_model_X_positions'])

        self.exit_lane_left_Z_range_end = data['exit_lane_left_Z_range_end']
        self.exit_lane_right_Z_range_end = data['exit_lane_right_Z_range_end']
        self.exit_lane_right_width = data['exit_lane_right_width']
        self.exit_lane_left_width = data['exit_lane_left_width']
        self.is_exit_lane_width = data['is_exit_lane_width']
        self.is_exit_Z_positions = np.asarray(data['is_exit_Z_positions'])
        self.exit_points_X = np.asarray(data['exit_points_X'])
        # self.exit_points_Z = np.asarray(data['exit_points_Z'])
        #
        # top_view_image.width_m = dict_to_dump['top_view_img_total_width_meters']
        # top_view_image.height_m = dict_to_dump['top_view_img_total_height_meters']
        # top_view_image.pixel_width = dict_to_dump['top_view_img_pixel_width_meters']
        # top_view_image.pixel_height = dict_to_dump['top_view_img_pixel_height_meters']
        #
        # front_view_image.camH = dict_to_dump['front_view_image_camera_height']
        # front_view_image.fl = dict_to_dump['front_view_image_focal_length']
        # front_view_image.x_center = dict_to_dump['front_view_image_x_center']
        # front_view_image.y_center = dict_to_dump['front_view_image_horizon']

        # with open(filename, 'w') as fp:
        #     json.dump(dict_to_dump, fp)

    @staticmethod
    def calc_model_from_next_right(next_right_model, dist, lane_model_Z_positions, degree=4):
        points = np.zeros((len(lane_model_Z_positions), 2))
        for i in range(len(lane_model_Z_positions)):
            nr_Z = lane_model_Z_positions[i]
            nr_X = next_right_model.Z2X(nr_Z)
            nr_dxdz = next_right_model.dX_dZ(nr_Z)
            alpha = np.math.atan(nr_dxdz)
            X_new = nr_X - np.math.cos(alpha) * dist
            Z_new = nr_Z + np.math.sin(alpha) * dist
            points[i, :] = [Z_new, X_new]

        lane_model = LaneModel()
        # lane_model.calc_model_from_points_least_square(points, degree=4)
        lane_model.calc_model_from_points_lagrange(points, degree)
        return lane_model

    def calc_center_lane_models(self):

        #self.leftmost_line_X = - (sum(self.lanes_widths[:-1]) + self.lanes_widths[-1] * 0.5)

        # self.main_lane_model.calc_model_from_points_lagrange(self.points)
        # print("lagrange model", self.main_lane_model.all())
        # create right most:
        right_most_main_road_lane = LaneModel()
        # right_most_main_road_lane.calc_model_from_points_least_square(self.points, degree=4)
        right_most_main_road_lane.calc_model_from_points_lagrange(self.points)
        #print("main model on zero Z", right_most_main_road_lane.Z2X(0))
        self.lane_models.append(right_most_main_road_lane)

        for lane_idx in range(1, self.num_main_lanes):
            lane_model = self.calc_model_from_next_right(self.lane_models[lane_idx-1],
                                                         0.5 * self.lanes_widths[lane_idx-1] + 0.5 * self.lanes_widths[lane_idx],
                                                         self.lane_model_Z_positions)
            self.lane_models.append(lane_model)

        # now let's shift all models such that the rightmost lane in main road will be on zero X:
        shift_model_a0 = -self.lane_models[0].Z2X(0)
        shift_model_a1 = -self.lane_models[0].a1

        for lane_model in self.lane_models:
            lane_model.shift_a0(shift_model_a0)
            lane_model.shift_a1(shift_model_a1)

        # print("least_square model", self.main_lane_model.all())

    @staticmethod
    def get_shifted_model_copy(model, shift):
        shifted_model = copy.copy(model)
        shifted_model.set_a0(shift)
        return shifted_model

    @staticmethod
    def left_line2_lane_idx(line_idx):
        # lane 0 - host usually - is mandated by line 1 on the left of it
        return line_idx - 1

    def build_main_road_line(self, line_idx, solidashed='dashed'):

        if line_idx == (self.num_lines -1):  # for leftmost lane:
            solidashed = 'solid'
        #  middle of scene is always the middle of the rightmost lane (with width self.lanes_widths[-1])
        if line_idx > 0:
            lane_idx = self.left_line2_lane_idx(line_idx)
            dist_from_model = self.lanes_widths[lane_idx] * 0.5
        else:
            lane_idx = 0
            dist_from_model = -self.lanes_widths[lane_idx] * 0.5
            solidashed = 'solid'

        line_model = self.calc_model_from_next_right(self.lane_models[lane_idx], dist_from_model, self.lane_model_Z_positions)
        width = self.lines_widths[line_idx]
        line = Line(lane_models=[line_model], Z_ranges=[self.lines_mark_z_ranges[line_idx, :]],
                    solidashed_types=[solidashed], widths=[width])
        return line

    def calc_exit_leftmost_model(self):
        exit_model = LaneModel()

        for i, Z_from_exit_beg in enumerate(self.is_exit_Z_positions):
            #Z = max(min(Z_from_exit_beg + self.is_exit_beg_position, self.lane_model_Z_positions[-1]), self.lane_model_Z_positions[0])
            Z = Z_from_exit_beg + self.is_exit_beg_position
            X_At_point_Z = self.lane_models[0].Z2X(Z) + self.is_exit_X_deltas_from_main[i]
            self.exit_points_X[i] = X_At_point_Z
            self.exit_points_Z[i] = Z
        exit_points = np.stack((self.exit_points_Z, self.exit_points_X)).T
        dX_dZ_at_Z_beg = self.lane_models[0].dX_dZ(self.is_exit_beg_position)
        exit_model.calc_quadratic_model_from_points_lagrange_fixed_dydx_at_x0(exit_points,
                                                                           dydx_at_x_dot=dX_dZ_at_Z_beg,
                                                                           x_dot=self.is_exit_beg_position)
        # exit_model.calc_model_from_points_least_square(exit_points)
        # # exit_model.add_model_3rd_4th_parameters(self.lane_models[0])
        # exit_model.calc_model_from_points_lagrange(exit_points)
        debug = False
        if (debug):
            import matplotlib.pyplot as plt
            em2 = LaneModel()
            ep = np.stack((self.is_exit_Z_positions, self.exit_points_X)).T
            lm0 = np.zeros((300, 2))
            em22 = np.zeros((300, 2))
            for i in range(-100,200):
                em22[i, :] = [i, exit_model.Z2X(i)]
                lm0[i, :] = [i, self.lane_models[0].Z2X(i)]
            plt.plot(exit_points[:, 0], exit_points[:, 1], 'ro', self.points[:, 0], self.points[:, 1], 'go', em22[:, 0],
                     em22[:, 1], 'r.', lm0[:, 0], lm0[:, 1], 'g-')
        return exit_model

    def calc_exit_lines_models(self):
        exit_lane_width = self.is_exit_lane_width#get_rand_range(3.2, 4.0)
        Z_points_to_calc_exits_with = [-100, 0, 100, 150, 200] # need 4 points for lagrange
        exit_model_left_line = self.calc_model_from_next_right(self.exit_model, (exit_lane_width * 0.5),
                                                               Z_points_to_calc_exits_with)
        exit_model_right_line = self.calc_model_from_next_right(self.exit_model, -(exit_lane_width * 0.5),
                                                                Z_points_to_calc_exits_with)

        return exit_model_left_line, exit_model_right_line

    def build_lines(self):
        assert self.num_main_lanes > 0
        # draw_line(self, lane_model, delta_X, width, color, Z_range):

        lines = list()
        # build all lines except for the main road rightmost lane:
        for i in range(1, self.num_lines):
            line = self.build_main_road_line(i)
            lines.append(line)
        # now build the last line, according to the scene type:
        right_most_lines = list()
        if self.is_exit_split_type == "no_exit":
            right_most_solid_lane_mark = self.build_main_road_line(line_idx=0, solidashed='solid')
            right_most_lines.append(right_most_solid_lane_mark)
        elif self.is_exit_split_type == "zero2one_unmarked_V":
            # first we make some calculations about the exit model:

            exit_model_left, exit_model_right = self.calc_exit_lines_models()
            # here we build the rightmost solid:
            idx = 0

            solidashed = ['solid', 'solid']
            widths = [self.lines_widths[idx], self.lines_widths[idx]]

            right_most_line_model = self.calc_model_from_next_right(self.lane_models[idx], -self.lanes_widths[idx] * 0.5,
                                                                    self.lane_model_Z_positions)
            # when building an exit lane you want it's beginning to
            rightmost_mismatch_2bfixed = right_most_line_model.Z2X(self.is_exit_beg_position) - exit_model_right.Z2X(self.is_exit_beg_position)
            exit_model_left.shift_a0(rightmost_mismatch_2bfixed)
            exit_model_right.shift_a0(rightmost_mismatch_2bfixed)
            dist2V = calc_Z_intersection_1_becomes_larget_than_2(exit_model_left, right_most_line_model, min_val=self.is_exit_beg_position)
            print("dist2V", dist2V)
            if dist2V is None:
                print("problem here with dist2v")
                return None
            rmlm_z_ranges = [[self.lines_mark_z_ranges[idx][0], self.is_exit_beg_position],
                        [dist2V, self.lines_mark_z_ranges[idx][1]]]
            lane_models = [right_most_line_model, right_most_line_model]

            main_lane_rightmost_solid = Line(lane_models=lane_models,
                                             Z_ranges=rmlm_z_ranges,
                                             solidashed_types=solidashed, widths=widths)

            # right_most_lines.append(main_lane_rightmost_solid)
            right_most_lines.append(main_lane_rightmost_solid)
            exit_lane_left_Z_range = np.stack((dist2V, self.exit_lane_left_Z_range_end)).T
            # now we build left exit line:
            exit_lane_left_solid = Line(lane_models=[exit_model_left],
                                        Z_ranges=[exit_lane_left_Z_range],
                                        solidashed_types=['solid'], widths=[self.exit_lane_left_width])
            # and the right exit line:
            exit_lane_right_Z_range = np.stack((self.is_exit_beg_position, self.exit_lane_right_Z_range_end)).T

            exit_lane_right_solid = Line(lane_models=[exit_model_right],
                                        Z_ranges=[exit_lane_right_Z_range],
                                        solidashed_types=['solid'], widths=[self.exit_lane_right_width])

            right_most_lines.append(exit_lane_left_solid)
            right_most_lines.append(exit_lane_right_solid)
        elif self.is_exit_split_type == 'one2one_dashed2solid2Y':
            # first we make some calculations about the exit model:
            exit_model_left, exit_model_right = self.calc_exit_lines_models()
            # here we build the rightmost solid:
            idx = 0

            solidashed = ['dashed', 'solid']
            widths = [self.lines_widths[idx], self.lines_widths[idx]]
            # todo: randomize different line widths

            right_most_line_model = self.calc_model_from_next_right(self.lane_models[idx],
                                                                    -self.lanes_widths[idx] * 0.5,
                                                                    self.lane_model_Z_positions)
            # when building an exit lane you want it's beginning to
            rightmost_mismatch_2bfixed = right_most_line_model.Z2X(self.is_exit_beg_position) - exit_model_right.Z2X(
                self.is_exit_beg_position)
            exit_model_left.shift_a0(rightmost_mismatch_2bfixed)
            exit_model_right.shift_a0(rightmost_mismatch_2bfixed)
            dist2V = calc_Z_intersection_1_becomes_larget_than_2(exit_model_left, right_most_line_model,
                                                                 min_val=self.is_exit_beg_position)
            print("dist2V", dist2V)
            if dist2V is None:
                print("problem here with dist2v")
                return None
            dashed_dist = get_rand_range(0, 50)

            rmlm_z_ranges = [[self.lines_mark_z_ranges[idx][0], dist2V-dashed_dist],
                             [dist2V-dashed_dist, self.lines_mark_z_ranges[idx][1]]]

            lane_models = [right_most_line_model, right_most_line_model]

            main_lane_rightmost_solid = Line(lane_models=lane_models,
                                             Z_ranges=rmlm_z_ranges,
                                             solidashed_types=solidashed, widths=widths)

            # right_most_lines.append(main_lane_rightmost_solid)
            right_most_lines.append(main_lane_rightmost_solid)
            exit_lane_left_Z_range = np.stack((dist2V, self.exit_lane_left_Z_range_end)).T
            # now we build left exit line:
            exit_lane_left_solid = Line(lane_models=[exit_model_left],
                                        Z_ranges=[exit_lane_left_Z_range],
                                        solidashed_types=['solid'], widths=[self.exit_lane_left_width])
            # and the right exit line:
            exit_lane_right_Z_range = np.stack((dist2V, self.exit_lane_right_Z_range_end)).T
            exit_lane_right_right_Z_range = np.stack((0, dist2V)).T
            exit_model_right_right = self.calc_model_from_next_right(self.lane_models[idx],
                                                                     - (self.lanes_widths[idx] * 0.5 + self.is_exit_lane_width),
                                                                     self.lane_model_Z_positions)
            exit_lane_right_solid = Line(lane_models=[exit_model_right_right, exit_model_right],
                                         Z_ranges=[exit_lane_right_right_Z_range, exit_lane_right_Z_range],
                                         solidashed_types=['solid', 'solid'], widths=[self.exit_lane_right_width, self.exit_lane_right_width])

            right_most_lines.append(exit_lane_left_solid)
            right_most_lines.append(exit_lane_right_solid)
        elif self.is_exit_split_type == 'one2two_unmarked_V':
            exit_model_left, exit_model_right = self.calc_exit_lines_models()
            # here we build the rightmost solid:
            idx = 0

            solidashed = ['dashed', 'solid']
            widths = [self.lines_widths[idx], self.lines_widths[idx]]
            # todo: randomize different line widths

            right_most_line_model = self.calc_model_from_next_right(self.lane_models[idx],
                                                                    -self.lanes_widths[idx] * 0.5,
                                                                    self.lane_model_Z_positions)
            # when building an exit lane you want it's beginning to
            rightmost_mismatch_2bfixed = right_most_line_model.Z2X(self.is_exit_beg_position) - exit_model_right.Z2X(
                self.is_exit_beg_position)
            exit_model_left.shift_a0(rightmost_mismatch_2bfixed)
            exit_model_right.shift_a0(rightmost_mismatch_2bfixed)
            dist2V = calc_Z_intersection_1_becomes_larget_than_2(exit_model_left, right_most_line_model,
                                                                 min_val=self.is_exit_beg_position)
            print("dist2V", dist2V)
            if dist2V is None:
                print("problem here with dist2v")
                return None
            dashed_dist = get_rand_range(0, 50)

            rmlm_z_ranges = [[self.lines_mark_z_ranges[idx][0], dist2V - dashed_dist],
                             [dist2V - dashed_dist, self.lines_mark_z_ranges[idx][1]]]

            lane_models = [right_most_line_model, right_most_line_model]

            main_lane_rightmost_solid = Line(lane_models=lane_models,
                                             Z_ranges=rmlm_z_ranges,
                                             solidashed_types=solidashed, widths=widths)

            # right_most_lines.append(main_lane_rightmost_solid)
            right_most_lines.append(main_lane_rightmost_solid)
            exit_lane_left_Z_range = np.stack((dist2V, self.exit_lane_left_Z_range_end)).T
            # now we build left exit line:
            exit_lane_left_solid = Line(lane_models=[exit_model_left],
                                        Z_ranges=[exit_lane_left_Z_range],
                                        solidashed_types=['solid'], widths=[self.exit_lane_left_width])
            # and the right exit line:
            exit_lane_right_Z_range = np.stack((dist2V, self.exit_lane_right_Z_range_end)).T
            exit_lane_right_right_Z_range = np.stack((0, dist2V)).T
            exit_model_right_right = self.calc_model_from_next_right(self.lane_models[idx],
                                                                     - (self.lanes_widths[
                                                                            idx] * 0.5 + self.is_exit_lane_width),
                                                                     self.lane_model_Z_positions)

            exit_lane_right_dashed = Line(lane_models=[exit_model_right_right, exit_model_right],
                                         Z_ranges=[exit_lane_right_right_Z_range, exit_lane_right_Z_range],
                                         solidashed_types=['dashed', 'dashed'],
                                         widths=[self.exit_lane_right_width, self.exit_lane_right_width])
            exit_lane_right_Z_range = np.stack((dist2V, self.exit_lane_right_Z_range_end)).T
            exit_lane_right_right_Z_range = np.stack((0, dist2V)).T
            exit_model_right_right = self.calc_model_from_next_right(self.lane_models[idx],
                                                                     - (self.lanes_widths[idx] * 0.5 + 2 * self.is_exit_lane_width),
                                                                     self.lane_model_Z_positions)
            # todo - randomize exit lane widths
            exit_lane_right_solid = Line(lane_models=[exit_model_right_right, exit_model_right],
                                         Z_ranges=[exit_lane_right_right_Z_range, exit_lane_right_Z_range],
                                         solidashed_types=['solid', 'solid'],
                                         widths=[self.exit_lane_right_width, self.exit_lane_right_width])

            right_most_lines.append(exit_lane_left_solid)
            right_most_lines.append(exit_lane_right_dashed)
            right_most_lines.append(exit_lane_right_solid)
        lines.extend(right_most_lines)

        return lines


if __name__ == "__main__":

    parent_dir_path = 'D:\phantomAI\data\synthesized_data'
    dt_folder_string = datetime.now().strftime("%Y_%m_%d__%H_%M_run")
    dir_path = os.path.join(parent_dir_path, dt_folder_string)
    fv_dir = os.path.join(dir_path, "front_view_image")
    tv_dir = os.path.join(dir_path, "top_view_image")
    md_dir = os.path.join(dir_path, "meta_data")

    try:
        os.mkdir(dir_path)
        os.mkdir(fv_dir)
        os.mkdir(tv_dir)
        os.mkdir(md_dir)
    except:
        pass
    for i in range(1000):
        lanes_factory = Lanes_factory()

        #lanes_factory.use_debug_set()
        #json_fp = 'D:\\phantomAI\\data\\synthesized_data\\2019_11_14_09_00_run\\meta_data\\meta_data_2019_11_14__09_00_28.json'
        json_fp = 'D:\\phantomAI\\data\\synthesized_data\\2019_11_23__22_15_run\\\meta_data\\meta_data_2019_11_23__22_15_42_866.json'
        lanes_factory.load_from_json(json_fp)


        lanes_factory.calc_center_lane_models()

        lines = lanes_factory.build_lines()



        dt_string = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        fvi_filename = "front_view_image_" + dt_string + ".png"
        tvi_filename = "top_view_image_" + dt_string + ".png"
        md_filename = "meta_data_" + dt_string + ".json"
        # fvi_dirname = "front_view_image"
        # tvi_dirname = 'top_view_image'
        # md_dirname = 'meta_data'
        fvi_fp = os.path.join(fv_dir, fvi_filename)
        tvi_fp = os.path.join(tv_dir, tvi_filename)
        md_fp = os.path.join(md_dir, md_filename)

        if lines is None:
            md_fp = os.path.join(md_dir, md_filename.replace("meta_data", "buggy_mita_deta"))
            lanes_factory.dump_meta_data(None, None, md_fp)
            continue
        tvi = TV_image()
        tvi.draw_lines(lines)
        #tvi.display()

        lrm = LinearRoadModel(0, 0)
        focal_length = get_rand_range(2000-20, 2000+20)
        camera_height = get_rand_range(1.2-0.1, 1.2 + 0.1)
        degrees_possible_pitch = 5
        degrees_possible_yaw = 5
        radians_possible_pitch = degrees_possible_pitch * np.pi / 180
        radians_possible_yaw = degrees_possible_yaw * np.pi / 180
        center_possible_pitch_pixels = np.math.sin(radians_possible_pitch) * focal_length
        center_possible_yaw_pixels = np.math.sin(radians_possible_yaw) * focal_length

        x_center = get_rand_range(959-center_possible_yaw_pixels, 959+center_possible_yaw_pixels)
        y_center = get_rand_range(603-center_possible_pitch_pixels, 603+center_possible_pitch_pixels)
        # print("x_center", x_center, "y_center", y_center,
        #       "    possible yaw, pitch pixels", center_possible_yaw_pixels, center_possible_pitch_pixels)
        fvi_factory = FVI_Factory(width_pix=1208, height_pix=1920,
                                  focal_length=focal_length, camera_height=camera_height,
                                  x_center=x_center, y_center=y_center)
        # vehicles = Vehicles()
        # vehicles.randomize(tvi, lrm)
        fvi = fvi_factory.draw_from_TV_image_and_linear_road_model(tvi, lrm)
        tvi.save(save_path=tvi_fp)
        fvi.save(save_path=fvi_fp)
        lanes_factory.dump_meta_data(tvi, fvi, md_fp)
        print("for the b point")