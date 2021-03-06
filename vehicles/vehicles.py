from tools.random_tools import get_rand_range, get_rand_out_of_list_item, get_rand_int
import random

# class vehicle:
#     def __init__(self, width, length, height, rear_side_X, rear_side_Y, rear_side_Z, yaw_angle):

vehicle_class_size_ranges = dict()

vehicle_class_size_ranges['mini'] = {'width': [1.6, 1.7], 'height': [1.4, 1.6], 'length': [4.0, 4.3]}
vehicle_class_size_ranges['sedan'] = {'width': [1.7, 1.9], 'height': [1.4, 1.8], 'length': [4.1, 4.5]}
vehicle_class_size_ranges['suv'] = {'width': [1.7, 2.2], 'height': [1.55, 2.4], 'length': [4.0, 4.8]}
vehicle_class_size_ranges['truck'] = {'width': [2.1, 2.5], 'height': [2.5, 3.0], 'length': [7.0, 14.0]}
vehicle_class_size_ranges['bus'] = {'width': [2.2, 2.6], 'height': [2.2, 3.3], 'length': [7.0, 11.0]}


class SingleVehicle:
    def __init__(self, distance, yaw_angle_drift, vcl_class, size, lane_idx, vcl_idx, visibility, position_in_lane):
        self.distance = distance
        self.yaw_angle_drift = yaw_angle_drift
        self.vcl_class = vcl_class
        self.size = size
        self.yaw = 0
        self.pitch = 0
        self.roll = 0

        self.lane_idx = lane_idx
        self.ID = vcl_idx
        self.visibility = visibility
        self.position_in_lane = position_in_lane


    def get_close_Z(self):
        # vcl starts at distance and ends at distance + length
        return self.distance

    def get_far_Z(self):
        # vcl starts at distance and ends at distance + length
        return self.distance + self.size['l']


class Vehicles:
    def __init__(self, num_lanes):
        self.num_lanes = num_lanes
        num_vehicles_per_lane_av = random.randint(0, 5)
        num_vehicles_per_lane_range = [max(0, num_vehicles_per_lane_av - 2), num_vehicles_per_lane_av + 2]
        self.num_vehicles_per_lane = get_rand_int(num_vehicles_per_lane_range[0], num_vehicles_per_lane_range[1], num_lanes)
        self.total_vehicles_num = sum(self.num_vehicles_per_lane)

        vehicles_distances = get_rand_range(10, 150, self.total_vehicles_num)
        vehicles_yaw_angle_drift = get_rand_range(10, 200, self.total_vehicles_num)
        vehicle_classes = get_rand_out_of_list_item(["mini", "sedan", "suv", "truck", "bus"],
                                                      num_items=self.total_vehicles_num,
                                                      objects_weights=[0.07, 0.4, 0.4, 0.07, 0.06])
        positions_in_lane = get_rand_range(-0.5, 0.5, self.total_vehicles_num)
        vehicles_sizes = self.init_vehicles_size(vehicle_classes)

        self.vehicles_objs = list()
        self.init_vehicles_objs(vehicles_distances, vehicles_yaw_angle_drift, vehicle_classes,
                                vehicles_sizes, positions_in_lane)
        self.vcls_distance_threshold = 2  # minimum of 2 meters between vehicles
        self.remove_overlapping_vehicles()

    def remove_vcl(self, vcl):

        self.num_vehicles_per_lane[vcl.lane_idx] -= 1
        self.total_vehicles_num -= 1
        self.vehicles_objs.remove(vcl)

    def init_vehicles_objs(self, vehicles_distances, vehicles_yaw_angle_drift, vehicle_classes,
                           vehicles_sizes, positions_in_lane):
        """
        Creates the list of SingleVehicle object in self.vehicles_objs
        """
        lanes_idxes = self.num_vehicles_per_lane.cumsum()
        lane_idx = 0
        vcl_id = 0
        for i in range(self.total_vehicles_num):
            while i >= lanes_idxes[lane_idx]:
                lane_idx += 1
            visibility = 'visible'
            if vehicles_distances[i] > 100:
                visibility = get_rand_out_of_list_item(['visible', 'non_visible'], objects_weights=[0.5, 0.5]
                                                       )
            vcl = SingleVehicle(vehicles_distances[i], vehicles_yaw_angle_drift[i], vehicle_classes[i],
                                vehicles_sizes[i], lane_idx, vcl_id, visibility, positions_in_lane[i])
            self.vehicles_objs.append(vcl)
            vcl_id += 1

    def get_vehicles_in_lane(self, lane_idx):
        vehicles_in_lane = list()
        for vcl_obj in self.vehicles_objs:
            if vcl_obj.lane_idx == lane_idx:
                vehicles_in_lane.append(vcl_obj)
        return vehicles_in_lane

    def remove_overlapping_vehicles(self):

        for lane_idx in range(self.num_lanes):
            vcls_in_lane = self.get_vehicles_in_lane(lane_idx)
            sorted_Z_vcls_in_lane = sorted(vcls_in_lane, key=lambda x: x.distance, reverse=False)
            prev_max_dist = 0
            for vcl in sorted_Z_vcls_in_lane:
                vcl_min_dist = vcl.get_close_Z()
                if vcl_min_dist < prev_max_dist:
                    self.remove_vcl(vcl)
                else:
                    prev_max_dist = vcl.get_far_Z() + self.vcls_distance_threshold

    def init_vehicles_size(self, vehicle_classes):
        vehicles_sizes = list()
        for i in range(self.total_vehicles_num):
            vcl_class = vehicle_classes[i]
            vcl_size = dict()
            vcl_size['w'] = get_rand_range(vehicle_class_size_ranges[vcl_class]['width'][0],
                                           vehicle_class_size_ranges[vcl_class]['width'][1])
            vcl_size['h'] = get_rand_range(vehicle_class_size_ranges[vcl_class]['height'][0],
                                           vehicle_class_size_ranges[vcl_class]['height'][1])
            vcl_size['l'] = get_rand_range(vehicle_class_size_ranges[vcl_class]['length'][0],
                                           vehicle_class_size_ranges[vcl_class]['length'][1])
            vehicles_sizes.append(vcl_size)
        return vehicles_sizes

    # def draw_single_vehicle_on_top_view(self, tvi, vcl, X, Z):

if __name__ == "__main__":
    vcls = Vehicles(num_lanes=3)
