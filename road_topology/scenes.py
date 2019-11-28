
class Scene:
    def __init__(self, main_road_lanes, scene_start):
        self.lanes = list()
        self.twiks = list()
        self.main_road_lanes = main_road_lanes
        self.scene_start = scene_start

    def get_lanes(self):
        return self.lanes


class Zero2oneExit(Scene):
    def __init__(self, ):

    self.