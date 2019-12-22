import numpy as np
from road_top_view_image.tv_image import TV_image
from road_manifold.linear_road_model import LinearRoadModel
from front_view_image.front_view_image import FVI_Factory
import os
from datetime import datetime
from tools.random_tools import get_rand_range
from vehicles.vehicles import Vehicles
from road_topology.lanes import LanesFactory
import copy

if __name__ == "__main__":
    parent_dir_path = 'D:\phantomAI\data\synthesized_data\\testing'
    dt_folder_string = datetime.now().strftime("%Y_%m_%d__%H_%M_run")
    dir_path = os.path.join(parent_dir_path, dt_folder_string)
    fv_dir = os.path.join(dir_path, "front_view_image")
    tv_dir = os.path.join(dir_path, "top_view_image")
    tv_v_dir = os.path.join(dir_path, "top_view_image_with_vehicles")
    md_dir = os.path.join(dir_path, "meta_data")

    try:
        os.mkdir(dir_path)
        os.mkdir(fv_dir)
        os.mkdir(tv_dir)
        os.mkdir(tv_v_dir)
        os.mkdir(md_dir)
    except:
        pass
    for i in range(10000):
        dt_string = datetime.now().strftime("%Y_%m_%d__%H_%M_%S_%f")[:-3]
        fvi_filename = "front_view_image_" + dt_string + ".png"
        seg_fvi_filename = "seg_front_view_image_" + dt_string + ".png"
        seg_cropped_fvi_filename = "seg_crop_front_view_image_" + dt_string + ".png"

        csv_filename = seg_fvi_filename.replace('seg', 'out').replace('png', 'csv')
        csv_crop_filename = seg_cropped_fvi_filename.replace('seg', 'out').replace('png', 'csv')

        tvi_filename = "top_view_image_" + dt_string + ".png"
        tvi_vehicles_filename = "top_view_image_vcls_" + dt_string + ".png"
        md_filename = "meta_data_" + dt_string + ".json"

        fvi_fp = os.path.join(fv_dir, fvi_filename)
        fvi_seg_fp = os.path.join(fv_dir, seg_fvi_filename)
        fvi_seg_cropped_fp = os.path.join(fv_dir, seg_cropped_fvi_filename)
        tvi_fp = os.path.join(tv_dir, tvi_filename)
        tvi_v_fp = os.path.join(tv_v_dir, tvi_vehicles_filename)
        md_fp = os.path.join(md_dir, md_filename)

        lanes_factory = LanesFactory()
        lines = lanes_factory.get_lane_marks()

        if lines is None:
            md_fp = os.path.join(md_dir, md_filename.replace("meta_data", "buggy_mita_deta"))
            lanes_factory.dump_meta_data(None, None, md_fp)
            continue

        vcls = Vehicles(num_lanes=lanes_factory.num_total_lanes)

        tvi = TV_image()
        tvi.draw_lines(lines)
        tvi_v = copy.deepcopy(tvi)
        tvi_v.draw_vehicles(vcls, lanes_factory.lane_models)

        # tvi.display()

        lrm = LinearRoadModel(0, 0)
        focal_length = get_rand_range(2000-20, 2000+20)
        camera_height = get_rand_range(1.2-0.1, 1.2 + 0.1)
        degrees_possible_pitch = 5
        degrees_possible_yaw = 10
        radians_possible_pitch = degrees_possible_pitch * np.pi / 180
        radians_possible_yaw = degrees_possible_yaw * np.pi / 180
        center_possible_pitch_pixels = np.math.sin(radians_possible_pitch) * focal_length
        center_possible_yaw_pixels = np.math.sin(radians_possible_yaw) * focal_length

        x_center = get_rand_range(959-center_possible_yaw_pixels, 959+center_possible_yaw_pixels)
        y_center = get_rand_range(603-center_possible_pitch_pixels, 603+center_possible_pitch_pixels)
        print("x_center", x_center, "y_center", y_center,
              "    possible yaw, pitch pixels", center_possible_yaw_pixels, center_possible_pitch_pixels)
        fvi_factory = FVI_Factory(width_pix=1208, height_pix=1920,
                                  focal_length=focal_length, camera_height=camera_height,
                                  x_center=x_center, y_center=y_center)
        # vehicles = Vehicles()
        # vehicles.randomize(tvi, lrm)
        fvi = fvi_factory.draw_from_TV_image_and_linear_road_model(tvi, lrm)
        fvi.draw_vehicles(vcls, lanes_factory.lane_models)
        fvi.calc_exit_merge()
        tvi.save(save_path=tvi_fp)
        tvi_v.save(save_path=tvi_v_fp)
        fvi.save(save_path=fvi_fp)
        fvi.save_seg_images(seg_img_save_path=fvi_seg_fp, cropped_img_save_path=fvi_seg_cropped_fp)
            # ,
            #                 csv_objs_save_path=csv_filename, csv_crop_objs_save_path=csv_crop_filename)
        lanes_factory.dump_meta_data(tvi, fvi, md_fp)
        print("for the b point")
