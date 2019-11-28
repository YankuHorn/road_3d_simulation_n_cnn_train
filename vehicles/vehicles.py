from tools.random_tools import get_rand_range, get_rand_out_of_list_item, get_rand_int
import random

# class vehicle:
#     def __init__(self, width, length, height, rear_side_X, rear_side_Y, rear_side_Z, yaw_angle):

vehicle_class_size_ranges = dict()

vehicle_class_size_ranges['mini'] = {'width': [1.6, 1.7], 'height': [1.4, 1.6], 'length': [4.0, 4.3]}
vehicle_class_size_ranges['sedan'] = {'width': [1.7, 1.9], 'height': [1.4, 1.6], 'length': [4.1, 4.5]}
vehicle_class_size_ranges['suv'] = {'width': [1.7, 2.0], 'height': [1.55, 2.0], 'length': [4.0, 4.8]}
vehicle_class_size_ranges['truck'] = {'width': [2.4, 2.5], 'height': [2.5, 3.0], 'length': [7.0, 14.0]}
vehicle_class_size_ranges['bus'] = {'width': [2.2, 2.55], 'height': [2.2, 3.0], 'length': [7.0, 11.0]}


class SingleVehicle:
    def __init__(self, distance, yaw_angle_drift, vcl_class, size, lane_idx):
        self.distance = distance
        self.yaw_angle_drift = yaw_angle_drift
        self.vcl_class = vcl_class
        self.size = size
        self.yaw = 0
        self.pitch = 0
        self.roll = 0

        self.lane_idx = lane_idx

class Vehicles:
    def __init__(self, num_lanes):
        self.num_vehicles_per_lane_av = random.randint(3, 5)
        self.num_vehicles_per_lane_range = [max(0, self.num_vehicles_per_lane_av - 3), self.num_vehicles_per_lane_av + 3]
        self.num_vehicles_per_lane = get_rand_int(self.num_vehicles_per_lane_range[0], self.num_vehicles_per_lane_range[1], num_lanes)
        self.total_vehicles_num = sum(self.num_vehicles_per_lane)
        self.vehicles_distances = get_rand_range(10, 200, self.total_vehicles_num)
        self.vehicles_yaw_angle_drift = get_rand_range(10, 200, self.total_vehicles_num)
        self.vehicle_classes = get_rand_out_of_list_item(["mini", "sedan", "suv", "truck", "bus"],
                                                         num_items=self.total_vehicles_num,
                                                         objects_weights=[0.07, 0.4, 0.4, 0.07, 0.06])

        self.vehicles_sizes = list()
        self.init_vehicles_size()

        self.vehicles_objs = list()
        self.init_vehicles_objs()

    def init_vehicles_objs(self):
        lane_counter = 0
        lanes_idxes = self.num_vehicles_per_lane.cumsum()
        lane_idx = 0
        for i in range(self.total_vehicles_num):
            while i >= lanes_idxes[lane_idx]:
                lane_idx += 1
            vcl = SingleVehicle(self.vehicles_distances[i], self.vehicles_yaw_angle_drift[i], self.vehicle_classes[i], self.vehicles_sizes[i], lane_idx)
            self.vehicles_objs.append(vcl)

    def init_vehicles_size(self):

        for i in range(self.total_vehicles_num):
            vcl_class = self.vehicle_classes[i]
            vcl_size = dict()
            vcl_size['w'] = get_rand_range(vehicle_class_size_ranges[vcl_class]['width'][0],
                                           vehicle_class_size_ranges[vcl_class]['width'][1])
            vcl_size['h'] = get_rand_range(vehicle_class_size_ranges[vcl_class]['height'][0],
                                           vehicle_class_size_ranges[vcl_class]['height'][1])
            vcl_size['l'] = get_rand_range(vehicle_class_size_ranges[vcl_class]['length'][0],
                                           vehicle_class_size_ranges[vcl_class]['length'][1])
            self.vehicles_sizes.append(vcl_size)

    # def draw_single_vehicle_on_top_view(self, tvi, vcl, X, Z):