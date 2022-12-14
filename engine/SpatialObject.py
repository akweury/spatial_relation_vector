# Created by shaji on 14-Dec-22

class SpatialObject():
    def __init__(self, color=None, shape=None, pos=None, size=None, material=None):
        self.color = color
        self.shape = shape
        self.pos = pos
        self.size = size
        self.material = material

    def print_info(self):
        print(f"color:{self.color}\n"
              f"color :{self.color}\n"
              f"shape :{self.shape}\n"
              f"pos :{self.pos}\n"
              f"size :{self.size}\n"
              f"material :{self.material}\n")
