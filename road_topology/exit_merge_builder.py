import numpy as np
from road_topology.lane_mark import LaneMark

# import random as random
# from road_top_view_image.tv_image import TV_image
# from road_manifold.linear_road_model import LinearRoadModel
# from front_view_image.front_view_image_factory import FVI_Factory
# import copy
# import os
# from datetime import datetime
# import json
from tools.random_tools import get_rand_out_of_list_item, get_rand_range, get_rand_list_item_simple
# from road_topology.lane_model import LaneModel


def calc_Z_intersection_1_becomes_larget_than_2(model1, model2, min_val=-100, max_val=200):
    enable_intersection_begin = False
    res = None
    for i in range(int(min_val), int(max_val)):
        X1 = model1.Z2X(i)
        X2 = model2.Z2X(i)
        if (X1 < X2) and (not enable_intersection_begin):
            enable_intersection_begin = True
        if enable_intersection_begin and (X1 > X2):
            res = i
            break
    return res


def get_zero2one_unmarked_V(lanes_factory):
    right_most_lines = list()
    exit_model_left, exit_model_right = lanes_factory.calc_exit_lines_models(lanes_factory.exit_model)
    # here we build the rightmost solid:
    idx = 0
    solidashed = ['solid', 'solid']
    widths = [lanes_factory.lane_marks_widths[idx], lanes_factory.lane_marks_widths[idx]]
    right_most_line_model = lanes_factory.calc_model_from_next_right(lanes_factory.lane_models[idx],
                                                            -lanes_factory.lanes_widths[idx] * 0.5,
                                                            lanes_factory.lane_model_Z_positions)
    # when building an exit lane you want it's beginning to
    rightmost_mismatch_2bfixed = right_most_line_model.Z2X(lanes_factory.is_exit_beg_position) - exit_model_right.Z2X(
        lanes_factory.is_exit_beg_position)

    exit_model_left.shift_a0(rightmost_mismatch_2bfixed)
    exit_model_right.shift_a0(rightmost_mismatch_2bfixed)
    dist2V = calc_Z_intersection_1_becomes_larget_than_2(exit_model_left, right_most_line_model,
                                                         min_val=lanes_factory.is_exit_beg_position,
                                                         max_val=lanes_factory.is_exit_beg_position+200)
    print("dist2V", dist2V)
    if dist2V is None:
        print("problem here with dist2v")
        return None
    rmlm_z_ranges = [[lanes_factory.lane_marks_mark_z_ranges[idx][0], lanes_factory.is_exit_beg_position],
                     [dist2V, lanes_factory.lane_marks_mark_z_ranges[idx][1]]]
    lane_models = [right_most_line_model, right_most_line_model]

    main_lane_rightmost_solid = LaneMark(lane_models=lane_models,
                                         Z_ranges=rmlm_z_ranges,
                                         solidashed_types=solidashed, widths=widths,
                                         host_lane_model=lanes_factory.lane_models[lanes_factory.host_lane_id])
    # right_most_lines.append(main_lane_rightmost_solid)
    right_most_lines.append(main_lane_rightmost_solid)
    exit_lane_left_Z_range = np.stack((dist2V, lanes_factory.exit_lane_left_Z_range_end)).T
    # now we build left exit line:
    exit_lane_left_solid = LaneMark(lane_models=[exit_model_left],
                                    Z_ranges=[exit_lane_left_Z_range],
                                    solidashed_types=['solid'], widths=[lanes_factory.exit_lane_left_width],
                                    host_lane_model=lanes_factory.lane_models[lanes_factory.host_lane_id],
                                    is_exit=[True])
    # and the right exit line:
    # exit_lane_right_Z_range = np.stack((lanes_factory.is_exit_beg_position, lanes_factory.exit_lane_right_Z_range_end)).T
    exit_lane_right_Z_range_preexit = np.stack(
        (lanes_factory.is_exit_beg_position, min(lanes_factory.is_exit_beg_position + 30, lanes_factory.exit_lane_right_Z_range_end))).T
    exit_lane_right_Z_range_postexit = np.stack(
        (min(lanes_factory.is_exit_beg_position + 30, lanes_factory.exit_lane_right_Z_range_end), lanes_factory.exit_lane_right_Z_range_end)).T
    exit_lane_right_solid = LaneMark(lane_models=[exit_model_right, exit_model_right],
                                     Z_ranges=[exit_lane_right_Z_range_preexit, exit_lane_right_Z_range_postexit],
                                     solidashed_types=['solid', 'solid'],
                                     widths=[lanes_factory.exit_lane_right_width, lanes_factory.exit_lane_right_width],
                                     host_lane_model=lanes_factory.lane_models[lanes_factory.host_lane_id],
                                    is_exit=[False, True])

    right_most_lines.append(exit_lane_left_solid)
    right_most_lines.append(exit_lane_right_solid)

    return right_most_lines


def get_one2one_dashed2solid2Y(lanes_factory):
    right_most_lines = list()
    # first we make some calculations about the exit model:
    exit_model_left, exit_model_right = lanes_factory.calc_exit_lines_models(lanes_factory.exit_model)
    # here we build the rightmost solid:
    idx = 0
    solidashed = ['dashed', 'solid']
    widths = [lanes_factory.lane_marks_widths[idx], lanes_factory.lane_marks_widths[idx]]
    # todo: randomize different line widths

    right_most_line_model = lanes_factory.calc_model_from_next_right(lanes_factory.lane_models[idx],
                                                            -lanes_factory.lanes_widths[idx] * 0.5,
                                                            lanes_factory.lane_model_Z_positions)
    # when building an exit lane you want it's beginning to
    rightmost_mismatch_2bfixed = right_most_line_model.Z2X(lanes_factory.is_exit_beg_position) - \
                                                           exit_model_right.Z2X(lanes_factory.is_exit_beg_position)

    exit_model_left.shift_a0(rightmost_mismatch_2bfixed)
    exit_model_right.shift_a0(rightmost_mismatch_2bfixed)
    dist2V = calc_Z_intersection_1_becomes_larget_than_2(exit_model_left, right_most_line_model,
                                                         min_val=lanes_factory.is_exit_beg_position ,
                                                         max_val=lanes_factory.is_exit_beg_position + 200)
    print("dist2V", dist2V)
    if dist2V is None:
        print("problem here with dist2v")
        return None
    dashed_dist = get_rand_range(0, 50)

    rmlm_z_ranges = [[lanes_factory.lane_marks_mark_z_ranges[idx][0], dist2V - dashed_dist],
                     [dist2V - dashed_dist, lanes_factory.lane_marks_mark_z_ranges[idx][1]]]

    lane_models = [right_most_line_model, right_most_line_model]

    main_lane_rightmost_solid = LaneMark(lane_models=lane_models,
                                         Z_ranges=rmlm_z_ranges,
                                         solidashed_types=solidashed, widths=widths,
                                         host_lane_model=lanes_factory.lane_models[lanes_factory.host_lane_id])

    # right_most_lines.append(main_lane_rightmost_solid)
    right_most_lines.append(main_lane_rightmost_solid)

    exit_lane_left_Z_range = np.stack((dist2V, lanes_factory.exit_lane_left_Z_range_end)).T
    # now we build left exit line:
    exit_lane_left_solid = LaneMark(lane_models=[exit_model_left],
                                    Z_ranges=[exit_lane_left_Z_range],
                                    solidashed_types=['solid'], widths=[lanes_factory.exit_lane_left_width],
                                    host_lane_model=lanes_factory.lane_models[lanes_factory.host_lane_id],
                                    is_exit=[True])
    # and the right exit line:
    exit_lane_right_Z_range = np.stack((dist2V, lanes_factory.exit_lane_right_Z_range_end)).T
    exit_lane_right_right_Z_range = np.stack((0, dist2V)).T
    exit_model_right_right = lanes_factory.calc_model_from_next_right(lanes_factory.lane_models[idx],
                                                             - (lanes_factory.lanes_widths[
                                                                    idx] * 0.5 + lanes_factory.is_exit_lane_width),
                                                             lanes_factory.lane_model_Z_positions)
    exit_lane_right_solid = LaneMark(lane_models=[exit_model_right_right, exit_model_right],
                                     Z_ranges=[exit_lane_right_right_Z_range, exit_lane_right_Z_range],
                                     solidashed_types=['solid', 'solid'],
                                     widths=[lanes_factory.exit_lane_right_width, lanes_factory.exit_lane_right_width],
                                     host_lane_model=lanes_factory.lane_models[lanes_factory.host_lane_id],
                                     is_exit=[False, True])

    right_most_lines.append(exit_lane_left_solid)
    right_most_lines.append(exit_lane_right_solid)
    return right_most_lines


def get_one2zero_unmarked_V_merge(lanes_factory):

    right_most_lines = list()
    merge_model_left, merge_model_right = lanes_factory.calc_exit_lines_models(lanes_factory.merge_model)

    # here we build the rightmost solid:
    idx = 0
    solidashed = ['solid', 'solid']
    widths = [lanes_factory.lane_marks_widths[idx], lanes_factory.lane_marks_widths[idx]]
    right_most_line_model = lanes_factory.calc_model_from_next_right(lanes_factory.lane_models[idx],
                                                            -lanes_factory.lanes_widths[idx] * 0.5,
                                                            lanes_factory.lane_model_Z_positions)
    # when building an merge lane you want it's beginning to
    rightmost_mismatch_2bfixed = right_most_line_model.Z2X(lanes_factory.is_merge_beg_position) - merge_model_right.Z2X(
        lanes_factory.is_merge_beg_position)

    merge_model_left.shift_a0(rightmost_mismatch_2bfixed)
    merge_model_right.shift_a0(rightmost_mismatch_2bfixed)

    dist2V = calc_Z_intersection_1_becomes_larget_than_2(right_most_line_model, merge_model_left,
                                                         min_val=lanes_factory.is_merge_beg_position - 300,
                                                         max_val=lanes_factory.is_merge_beg_position)
    print("dist2V", dist2V)
    if dist2V is None:
        rmlm_z_ranges = [[lanes_factory.lane_marks_mark_z_ranges[idx][0], lanes_factory.lane_marks_mark_z_ranges[idx][1]]]
        lane_models = [right_most_line_model]
        main_lane_rightmost_solid = LaneMark(lane_models=lane_models,
                                             Z_ranges=rmlm_z_ranges,
                                             solidashed_types=solidashed, widths=widths,
                                             host_lane_model=lanes_factory.lane_models[lanes_factory.host_lane_id])
        right_most_lines.append(main_lane_rightmost_solid)
    else:
        rmlm_z_ranges = [[lanes_factory.lane_marks_mark_z_ranges[idx][0], dist2V],
                         [lanes_factory.is_merge_beg_position, lanes_factory.lane_marks_mark_z_ranges[idx][1]]]
        lane_models = [right_most_line_model, right_most_line_model]

        main_lane_rightmost_solid = LaneMark(lane_models=lane_models,
                                             Z_ranges=rmlm_z_ranges,
                                             solidashed_types=solidashed, widths=widths,
                                             host_lane_model=lanes_factory.lane_models[lanes_factory.host_lane_id])

        right_most_lines.append(main_lane_rightmost_solid)
        # finished with right most lane

        merge_lane_left_Z_range = np.stack((lanes_factory.merge_lane_left_Z_range_begin, dist2V)).T
        # now we build left merge line:
        merge_lane_left_solid = LaneMark(lane_models=[merge_model_left],
                                        Z_ranges=[merge_lane_left_Z_range],
                                        solidashed_types=['solid'], widths=[lanes_factory.exit_lane_left_width],
                                        host_lane_model=lanes_factory.lane_models[lanes_factory.host_lane_id],
                                        is_merge=[True, True])
        # and the right exit line:
        merge_lane_right_Z_range = np.stack((lanes_factory.merge_lane_right_Z_range_begin, lanes_factory.is_merge_beg_position)).T

        merge_lane_right_solid = LaneMark(lane_models=[merge_model_right],
                                         Z_ranges=[merge_lane_right_Z_range],
                                         solidashed_types=['solid'], widths=[lanes_factory.exit_lane_right_width],
                                         host_lane_model=lanes_factory.lane_models[lanes_factory.host_lane_id],
                                         is_merge=[True, True])

        right_most_lines.append(merge_lane_left_solid)
        right_most_lines.append(merge_lane_right_solid)

    return right_most_lines


# elif self.is_exit_split_type == 'one2two_unmarked_V':
        #     exit_model_left, exit_model_right = self.calc_exit_lines_models()
        #     # here we build the rightmost solid:
        #     idx = 0
        #
        #     solidashed = ['dashed', 'solid']
        #     widths = [self.lane_marks_widths[idx], self.lane_marks_widths[idx]]
        #     # todo: randomize different line widths
        #
        #     right_most_line_model = self.calc_model_from_next_right(self.lane_models[idx],
        #                                                             -self.lanes_widths[idx] * 0.5,
        #                                                             self.lane_model_Z_positions)
        #     # when building an exit lane you want it's beginning to
        #     rightmost_mismatch_2bfixed = right_most_line_model.Z2X(self.is_exit_beg_position) - exit_model_right.Z2X(
        #         self.is_exit_beg_position)
        #     exit_model_left.shift_a0(rightmost_mismatch_2bfixed)
        #     exit_model_right.shift_a0(rightmost_mismatch_2bfixed)
        #     dist2V = calc_Z_intersection_1_becomes_larget_than_2(exit_model_left, right_most_line_model,
        #                                                          min_val=self.is_exit_beg_position)
        #     print("dist2V", dist2V)
        #     if dist2V is None:
        #         print("problem here with dist2v")
        #         return None
        #     dashed_dist = get_rand_range(0, 50)
        #
        #     rmlm_z_ranges = [[self.lane_marks_mark_z_ranges[idx][0], dist2V - dashed_dist],
        #                      [dist2V - dashed_dist, self.lane_marks_mark_z_ranges[idx][1]]]
        #
        #     lane_models = [right_most_line_model, right_most_line_model]
        #
        #     main_lane_rightmost_solid = LaneMark(lane_models=lane_models,
        #                                          Z_ranges=rmlm_z_ranges,
        #                                          solidashed_types=solidashed, widths=widths)
        #
        #     # right_most_lines.append(main_lane_rightmost_solid)
        #     right_most_lines.append(main_lane_rightmost_solid)
        #     exit_lane_left_Z_range = np.stack((dist2V, self.exit_lane_left_Z_range_end)).T
        #     # now we build left exit line:
        #     exit_lane_left_solid = LaneMark(lane_models=[exit_model_left],
        #                                     Z_ranges=[exit_lane_left_Z_range],
        #                                     solidashed_types=['solid'], widths=[self.exit_lane_left_width])
        #     # and the right exit line:
        #     exit_lane_right_Z_range = np.stack((dist2V, self.exit_lane_right_Z_range_end)).T
        #     exit_lane_right_right_Z_range = np.stack((0, dist2V)).T
        #     exit_model_right_right = self.calc_model_from_next_right(self.lane_models[idx],
        #                                                              - (self.lanes_widths[
        #                                                                     idx] * 0.5 + self.is_exit_lane_width),
        #                                                              self.lane_model_Z_positions)
        #
        #     exit_lane_right_dashed = LaneMark(lane_models=[exit_model_right_right, exit_model_right],
        #                                       Z_ranges=[exit_lane_right_right_Z_range, exit_lane_right_Z_range],
        #                                       solidashed_types=['dashed', 'dashed'],
        #                                       widths=[self.exit_lane_right_width, self.exit_lane_right_width])
        #     exit_lane_right_Z_range = np.stack((dist2V, self.exit_lane_right_Z_range_end)).T
        #     exit_lane_right_right_Z_range = np.stack((0, dist2V)).T
        #     exit_model_right_right = self.calc_model_from_next_right(self.lane_models[idx],
        #                                                              - (self.lanes_widths[
        #                                                                     idx] * 0.5 + 2 * self.is_exit_lane_width),
        #                                                              self.lane_model_Z_positions)
        #     # todo - randomize exit lane widths
        #     exit_lane_right_solid = LaneMark(lane_models=[exit_model_right_right, exit_model_right],
        #                                      Z_ranges=[exit_lane_right_right_Z_range, exit_lane_right_Z_range],
        #                                      solidashed_types=['solid', 'solid'],
        #                                      widths=[self.exit_lane_right_width, self.exit_lane_right_width])
        #
        #     right_most_lines.append(exit_lane_left_solid)
        #     right_most_lines.append(exit_lane_right_dashed)
        #     right_most_lines.append(exit_lane_right_solid)