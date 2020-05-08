import math

SCENE_SIZE = 8
PIXELS_PER_METER = 8
PIXEL_FRACTION = 1 / PIXELS_PER_METER
MAP_SIZE = SCENE_SIZE * PIXELS_PER_METER
PUPPER_HEIGHT = 0.3
WALL_HEIGHT = 8
WALL_THICKNESS = 1
MAX_WALL_INNESS = 2.5

IMAGE_WIDTH = 128# 192
IMAGE_HEIGHT = 128# 108

SHOTS_PER_SCENE = 64
VIEW_DIM = 7

HEIGHT_VARIATION = 0.1
PITCH_VARIATION = 20 * math.pi / 180
ROLL_VARIATION = 15 * math.pi / 180