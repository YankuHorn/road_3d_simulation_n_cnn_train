import numpy as np

from road_topology.lane_model import LaneModel
from tools.random_tools import get_rand_out_of_list_item, get_rand_range

class LaneMark:
    def __init__(self, lane_models, Z_ranges, solidashed_types, widths, host_lane_model, is_exit=None, is_merge=None):

        self.num_segments = len(lane_models)
        self.lane_models = lane_models

        self.solidashed_types = solidashed_types
        self.widths = widths

        self.Z_ranges, self.visibilities = self.calc_visibility_and_z_range(host_lane_model, Z_ranges)
        self.gap = self.calc_gaps()

        if is_exit is None:
            self.is_exit = list()
            for i in range(self.num_segments):
                self.is_exit.append(False)
        else:
            self.is_exit = is_exit
        if is_merge is None:
            self.is_merge = list()
            for i in range(self.num_segments):
                self.is_merge.append(False)
        else:
            self.is_merge = is_merge

    def calc_gaps(self):

        has_gap = get_rand_out_of_list_item([True, False], objects_weights=[0.05, 0.95])
        gap = dict()
        if has_gap:
            gap['begin'] = get_rand_range(0, 100)
            gap['length'] = get_rand_range(5, 30)
        else:
            gap['begin'] = 0  # get_rand_range(0, 100)
            gap['length'] = 0  # get_rand_range(5, 30)
        return gap

    def calc_visibility_and_z_range(self, host_model, Z_ranges):

        z_ranges_list = list()
        visibilities = list()
        for i, lane_model in enumerate(self.lane_models):
            visibility = None
            model_Z_range = np.asarray([0, 1000])

            dist2host_0 = abs(host_model.Z2X(0) - lane_model.Z2X(0))
            dist2host_100 = abs(host_model.Z2X(100) - lane_model.Z2X(100))

            if (dist2host_0 < 6) and (dist2host_100 < 6):
                visibility = get_rand_out_of_list_item(["visible", 'not_visible'], objects_weights=[0.99, 0.01])
            elif (dist2host_0 < 6) and (dist2host_100 > 6):  # split lane I'd assume
                visibility = get_rand_out_of_list_item(["barely_visible_in_dist", 'not_visible'], objects_weights=[0.9, 0.1])
            elif (dist2host_0 > 6) and (dist2host_100 < 6):  # merge
                visibility = get_rand_out_of_list_item(["barely_visible_in_near", 'not_visible'],
                                                       objects_weights=[0.9, 0.1])
            elif (dist2host_0 > 6) and (dist2host_100 > 6):
                visibility = get_rand_out_of_list_item(["barely_visible", 'not_visible'],
                                                       objects_weights=[0.1, 0.9])
            if visibility == "not_visible":
                z_ranges_list.append(np.asarray([0, 0]))
                continue
            elif visibility == "barely_visible_in_dist":
                model_Z_range = np.stack((get_rand_range(0, 20),
                                          get_rand_range(50, 100))).T

            elif visibility == "barely_visible_in_near":
                model_Z_range = np.stack((get_rand_range(10, 60),
                                    get_rand_range(80, 150))).T
            elif visibility == "barely_visible":
                model_Z_range = np.stack((get_rand_range(30, 60),
                                    get_rand_range(60, 90))).T
            elif visibility == "visible":
                model_Z_range = Z_ranges[i]
            model_Z_range = np.asarray([max(model_Z_range[0], Z_ranges[i][0]),
                                        min(model_Z_range[1], Z_ranges[i][1])])

            z_ranges_list.append(model_Z_range)
            visibilities.append(visibility)
        return z_ranges_list, visibilities

