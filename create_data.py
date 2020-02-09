import numpy as np
import matplotlib.pyplot as plt

from road_manifold.road_manifold import RoadManifold
from road_manifold.road_manifold import RoadManifoldsFactory
from front_view_image.front_view_image import fvi_factory, FV_image
from road_top_view_image.tv_image import TV_image


if __name__ == "__main__":
    tv = TV_image()

    tv.draw_lanes()
    tv.display()

    range_mean = [[10, 50], [10, 50]]  # -150 150 -150
    range_std = [[50, 50], [50, 50]]  # 25 25 250 250
    range_num_gaussians = [1, 1]  # 1 7
    range_magnitude = [10, 10]  # 0 50
    range_angle = [0, 0]  # 0 90

    rmf = RoadManifoldsFactory(range_mean, range_std, range_num_gaussians, range_magnitude, range_angle)
    road_manifold = rmf.get_manifold()
    road_manifold.create_point_cloud()
    plt.imshow(road_manifold.pc.pc)

    width_pix = 480
    height_pix = 320
    focal_length = 350
    camera_height = 1.5

    fvif = fvi_factory(width_pix, height_pix, focal_length, camera_height)
    fv = fvif.draw_from_TV_image_and_manifold(tv, road_manifold)

