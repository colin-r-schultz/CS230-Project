import pybullet as p
import pybullet_data
import math
import time
import random
import numpy as np
import pkgutil
import faulthandler
import sys

faulthandler.enable()

DATASET_PATH = 'datasets'
if len(sys.argv) > 1:
    DATASET_PATH = sys.argv[1]

SCENE_SIZE = 8
PIXELS_PER_METER = 8
PIXEL_FRACTION = 1 / PIXELS_PER_METER
MAP_SIZE = SCENE_SIZE * PIXELS_PER_METER
PUPPER_HEIGHT = 0.3
WALL_HEIGHT = 8
WALL_THICKNESS = 1
MAX_WALL_INNESS = 2.5

IMAGE_WIDTH = 192
IMAGE_HEIGHT = 108

SHOTS_PER_SCENE = 64
VIEW_DIM = 7

HEIGHT_VARIATION = 0.1
PITCH_VARIATION = 20 * math.pi / 180
ROLL_VARIATION = 15 * math.pi / 180

UNIT_SCALE = {
    'meters': 1,
    'centimeters': 0.01,
    'millimeters': 0.001,
    'inches': 0.0254
}

model_names = []
visual_shape_dict = {}
collision_shape_dict = {}
units_dict = {}
pixel_collision = -1
wall_collision = -1
pixel_extents = np.array([PIXEL_FRACTION, PIXEL_FRACTION, PUPPER_HEIGHT])


def load_all_models():
    global pixel_collision, wall_collision
    orient = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
    extents = pixel_extents * 0.5
    pixel_collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=extents, collisionFramePosition=extents)
    wall_extents = np.array([SCENE_SIZE, WALL_THICKNESS, WALL_HEIGHT]) * 0.5
    wall_offsets = wall_extents.copy()
    wall_offsets[1] = 0
    wall_collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=wall_extents, collisionFramePosition=wall_offsets)
    dictfile = open('models/objlist.txt')
    for line in dictfile:
        split = line[:-1].split(' ')
        assert(len(split) == 3)
        objname = split[0]
        units = split[1]
        scale = [UNIT_SCALE[units]] * 3
        model_names.append(objname)
        units_dict[objname] = units
        pos = [float(x) for x in split[2].split(',')]
        visual_shape_dict[objname] = p.createVisualShape(shapeType=p.GEOM_MESH, fileName='models/' + objname + '.obj',
            visualFrameOrientation=orient, visualFramePosition=pos, meshScale=scale)
        collision_shape_dict[objname] = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName='models/' + objname + '.obj',
            collisionFrameOrientation=orient, collisionFramePosition=pos, meshScale=scale)

def insert_model(model, pos=(0, 0), rot=0):
    pos = [pos[0], pos[1], 0]
    orient = p.getQuaternionFromEuler([0, 0, rot])
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape_dict[model], baseVisualShapeIndex=visual_shape_dict[model],
        basePosition=pos, baseOrientation=orient)

def random_pos(walls):
    x = walls[0] + random.random() * (SCENE_SIZE - walls[2] - walls[0])
    y = walls[1] + random.random() * (SCENE_SIZE - walls[3] - walls[1])
    return np.array([x, y])


def random_color():
    return [random.random(), random.random(), random.random(), 1]

def colorize(obj):
    p.changeVisualShape(obj, -1, rgbaColor=random_color())

def get_forward_vector(rot):
    forward_vec = np.ndarray((3,))
    forward_vec[0] = math.cos(rot[0]) * math.cos(rot[1])
    forward_vec[1] = math.sin(rot[0]) * math.cos(rot[1])
    forward_vec[2] = math.sin(rot[1])
    return forward_vec

def build_scene():
    map_label = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    bodies = []

    walls = np.random.sample((4,)) * MAX_WALL_INNESS
    wall_pixels = np.ceil(walls / PIXEL_FRACTION).astype(np.int)
    map_label[wall_pixels[0]:MAP_SIZE-wall_pixels[2],wall_pixels[1]:MAP_SIZE-wall_pixels[3]] = 1
    orients = [p.getQuaternionFromEuler([0, 0, math.pi / 2]),
        p.getQuaternionFromEuler([0, 0, 0])]
    for i in range(4):
        offset = walls[i] - WALL_THICKNESS / 2
        if i > 1:
            offset = SCENE_SIZE - offset
        pos = np.zeros((3,))
        pos[i % 2] = offset
        obj = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision, baseOrientation=orients[i % 2],
            basePosition=pos)
        bodies.append(obj)
        colorize(obj)

    
    for i in range(random.randint(6, 10)):
        obj = insert_model(random.choice(model_names), pos=random_pos(walls), rot=random.random() * 2 * math.pi)
        colorize(obj)
        bodies.append(obj)

    # for i in range(8):
    #     p.addUserDebugLine([0, 0, i / 4], [SCENE_SIZE, 0, i / 4], [0, 0, 1])
    #     p.addUserDebugLine([SCENE_SIZE, 0, i / 4], [SCENE_SIZE, SCENE_SIZE, i / 4], [0, 0, 1])
    #     p.addUserDebugLine([SCENE_SIZE, SCENE_SIZE, i / 4], [0, SCENE_SIZE, i / 4], [0, 0, 1])
    #     p.addUserDebugLine([0, SCENE_SIZE, i / 4], [0, 0, i / 4], [0, 0, 1])

    return walls, map_label, bodies
    

def get_map(map_label):
    for x in range(MAP_SIZE):
        for y in range(MAP_SIZE):
            if map_label[x, y] == 0:
                continue
            aabbMin = np.array([PIXEL_FRACTION * x, PIXEL_FRACTION * y, 0])
            aabbMax = aabbMin + pixel_extents
            objs = p.getOverlappingObjects(aabbMin, aabbMax)
            if objs is None or len(objs) == 0:
                continue
            box = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=pixel_collision, basePosition=aabbMin)
            for obj in objs:
                pts = p.getClosestPoints(box, obj[0], 0.0)
                if pts is not None and len(pts) > 0:
                    map_label[x, y] = 0
                    break
            
            p.removeBody(box)
        # print('Row {} done'.format(x))
            
    return map_label

def pack_viewpoint(pos, rot):
    view = np.ndarray((VIEW_DIM,))
    view[:3] = pos
    view[3:6] = get_forward_vector(rot)
    view[6] = rot[2]
    return view

def get_image(view):
    roll = view[6]
    pos = view[:3]
    forward_vec = view[3:6]
    target_pos = pos + forward_vec
    up_vec = -forward_vec[2] * forward_vec
    up_vec[2] += 1
    up_vec /= np.linalg.norm(up_vec)
    right_vec = np.cross(forward_vec, up_vec)
    roll_vec = up_vec * math.cos(roll) + right_vec * math.sin(roll)
    vm = p.computeViewMatrix(pos, target_pos, roll_vec)
    pm = p.computeProjectionMatrixFOV(69.4, IMAGE_WIDTH / IMAGE_HEIGHT, 0.01, 2 * SCENE_SIZE)
    return p.getCameraImage(IMAGE_WIDTH, IMAGE_HEIGHT, vm, pm)

def process_scene():
    walls, map_label, bodies = build_scene()
    map_label = get_map(map_label)
    orient = p.getQuaternionFromEuler([0, 0, random.random() * 2 * math.pi])
    planeId = p.loadURDF("plane.urdf", baseOrientation=orient)
    colorize(planeId)
    bodies.append(planeId)
    shots = 0
    views = np.ndarray((SHOTS_PER_SCENE, VIEW_DIM), dtype=np.float32)
    images = np.ndarray((SHOTS_PER_SCENE, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    while shots < SHOTS_PER_SCENE:
        rand = random_pos(walls)
        coords = np.floor(rand / PIXEL_FRACTION).astype(np.int)
        if map_label[coords[0], coords[1]] == 0:
            continue
        pos = np.array([rand[0], rand[1], PUPPER_HEIGHT - random.random() * HEIGHT_VARIATION])
        pitch = (1 - 2 * random.random()) * PITCH_VARIATION
        roll = (1 - 2 * random.random()) * ROLL_VARIATION
        rot = np.array([random.random() * 2 * math.pi, pitch, roll])
        view = pack_viewpoint(pos, rot)
        
        w, h, rgb, depth, seg = get_image(view)
        images[shots, :, :, :] = rgb[:,:,:3]
        # plt.imshow(rgb)
        # plt.show()
        views[shots, :] = view
        shots += 1
    for obj in bodies:
        p.removeBody(obj)
    return images, views, map_label
        
def add_axis(arr):
    return arr.reshape((1,) + arr.shape)

client = p.connect(p.DIRECT)
# egl = pkgutil.get_loader('eglRenderer')
# plugin = None
# if egl is not None:
#     plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
#     print('Using eglRenderer')
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
load_all_models()

print("Starting...")
M = 40000
CHUNK_SIZE = 200
N_CHUNKS = M // CHUNK_SIZE
start = time.time()
for j in range(N_CHUNKS): 
    image_dataset = np.zeros((CHUNK_SIZE, SHOTS_PER_SCENE, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    view_dataset = np.zeros((CHUNK_SIZE, SHOTS_PER_SCENE, VIEW_DIM), dtype=np.float32)
    map_dataset = np.zeros((CHUNK_SIZE, MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    for i in range(CHUNK_SIZE):
        images, views, map_label =  process_scene()
        image_dataset[i] = images
        view_dataset[i] = views
        map_dataset[i] = map_label
    np.save(DATASET_PATH + '/image_dataset.npy', image_dataset)
    np.save(DATASET_PATH + '/view_dataset.npy', view_dataset)
    np.save(DATASET_PATH + '/map_dataset.npy', map_dataset)
    print("Saving chunk {} ({} total scenes). {} seconds elapsed.".format(j+1, (j+1) * CHUNK_SIZE, time.time() - start))
print("TIME:", time.time() - start)

# if plugin is not None:
#     p.unloadPlugin(plugin)
p.disconnect()
