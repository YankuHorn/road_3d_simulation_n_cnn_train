import numpy as np
import os
from datetime import datetime
import json


def deg2rad(deg):
    rad = deg*np.pi/180
    return rad

class LinearRoadModel:
    def __init__(self, pitch, roll):
        self.pitch_deg = pitch
        self.pitch_rad = deg2rad(pitch)
        # self.yaw = yaw
        self.roll_deg = roll
        self.roll_rad = deg2rad(roll)

    def load_model_from_file(self, filename):
        data = json.load(filename)
        self.pitch_deg = data['pitch_deg']
        self.pitch_rad = deg2rad(self.pitch_deg)
        # self.yaw = data['yaw']
        self.roll_deg = data['roll_deg']
        self.roll_rad = deg2rad(self.roll_deg)

    def log_model_to_file(self, filename):
        data_dict = {'pitch_deg': self.pitch_deg, 'roll_deg': self.roll_deg}
        with open(filename, "w") as write_file:
            json.dump(data_dict, write_file)


class LinearRoadModelFactory:
    def __init__(self, range_pitch_deg, range_yaw_deg, range_roll_deg):
        self.pitch_range_deg = range_pitch_deg
        self.yaw_range_deg = range_yaw_deg
        self.roll_range_deg = range_roll_deg
    def get_model(self):
        pitch_deg = self.pitch_range_deg[0] + np.random.rand() * (self.pitch_range_deg[1] - self.pitch_range_deg[0])
        yaw_deg = self.yaw_range_deg[0] + np.random.rand() * (self.yaw_range_deg[1] - self.yaw_range_deg[0])
        roll_deg = self.roll_range_deg[0] + np.random.rand() * (self.roll_range_deg[1] - self.roll_range_deg[0])
        linear_road_model = LinearRoadModel(pitch_deg, yaw_deg, roll_deg)
        return linear_road_model


if __name__ == "__main__":
    range_pitch_deg = [-10, 10]
    range_yaw_deg = [-20, 20]
    range_roll_deg = [-1, 1]
    new_rm = True
    if new_rm:
        lrmf = LinearRoadModelFactory(range_pitch_deg, range_yaw_deg, range_roll_deg)
        linear_road_model_ = lrmf.get_model()

        dt_string = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        dir_path = 'D:\phantomAI\data\synthesized_data'
        filename_ = 'linear_road_model_' + dt_string
        full_path = os.path.join(dir_path, filename_)
        linear_road_model_.log_model_to_file(full_path)

    # plt.imshow(road_manifold.rm_pc.point_cloud[:, :, 1])
    # plt.show()
    # print("for the breakpoint")