import numpy as np
import pybullet as p
from constants import *
import scene_builder
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf
import architectures.recurrent_gru as model
import time
# np.random.seed(12345)

walls, map_label, bodies = scene_builder.build_scene_full()

STEP_SIZE = PIXEL_FRACTION
ROBOT_RADIUS = PIXEL_FRACTION / 2
NUM_TRIES = 64
MAX_DIST = SCENE_SIZE * np.sqrt(2)
MAX_STEPS = 300
EMBEDDING_SIZE = 512

def pos_to_coords(pos):
    coords = np.floor(pos / PIXEL_FRACTION).astype(np.int)
    return (coords[0], coords[1])

def random_valid_pos(walls, map_label):
    while True:
        pos = scene_builder.random_pos(walls)
        if map_label[pos_to_coords(pos)] == 1:
            return pos

frames = []

def build_model():
    def representation_network(pretrained=False):
        img_input = tf.keras.Input([None] + IMG_SHAPE)
        pose_input = tf.keras.Input([None, VIEW_DIM])

        seq_length = 16

        reshaped_img_input = tf.reshape(img_input, [-1] + IMG_SHAPE)
        reshaped_pose_input = tf.reshape(pose_input, [-1, VIEW_DIM])

        conv = model.TileConvNet(VIEW_DIM, EMBEDDING_SIZE, pretrained)

        features = conv([reshaped_img_input, reshaped_pose_input])

        features = tf.reshape(features, [-1, seq_length, EMBEDDING_SIZE])
        gru = tf.keras.layers.GRU(EMBEDDING_SIZE)
        embedding = gru(features)

        repmodel = tf.keras.Model([img_input, pose_input], embedding, name='representation_net')
        return repmodel, conv, gru

    img_input = tf.keras.Input([None] + IMG_SHAPE)
    pose_input = tf.keras.Input([None, VIEW_DIM])

    rep_net, conv, gru = representation_network() 
    embedding = rep_net([img_input, pose_input])
    mapping_net = model.mapping_network()
    map_estimate = mapping_net(embedding)

    e2e_model = tf.keras.Model([img_input, pose_input], map_estimate, name='e2e_model')
    e2e_model.load_weights('checkpoints/' + model.CHECKPOINT_PATH + '/e2e_model_119')

    img_input2 = tf.keras.Input(IMG_SHAPE)
    pose_input2 = tf.keras.Input([VIEW_DIM])
    features = conv([img_input2, pose_input2])
    prev_embedding = tf.keras.Input([EMBEDDING_SIZE])
    embedding = gru(tf.reshape(features, [-1, 1, EMBEDDING_SIZE]), initial_state=prev_embedding)
    rep_net2 = tf.keras.Model([img_input2, pose_input2, prev_embedding], embedding, name='rep_net')
    return rep_net2, mapping_net

def prepocess_image(img):
    img = img[:,:,:3].astype(np.float) / 255
    img = img * 2 - 1
    return img.reshape([1] + IMG_SHAPE)

def normalize_view(view):
    view = (view - VIEW_MEANS) / VIEW_VARIATION
    return view.reshape((1, VIEW_DIM))


def get_image(pos, rot):
    pos = np.array([pos[0], pos[1], PUPPER_HEIGHT - np.random.random() * HEIGHT_VARIATION])
    pitch = (1 - 2 * np.random.random()) * PITCH_VARIATION
    roll = (1 - 2 * np.random.random()) * ROLL_VARIATION
    rot = np.array([rot, pitch, roll])
    view = scene_builder.pack_viewpoint(pos, rot)
    w, h, rgb, depth, seg = scene_builder.get_image(view)
    return prepocess_image(rgb), normalize_view(view)

rep_net, map_net = build_model()

goal = random_valid_pos(walls, map_label)
position = random_valid_pos(walls, map_label)

start = time.time()

embedding = np.zeros([1, EMBEDDING_SIZE])
img, pose = get_image(position, np.random.random() * 2 * np.pi)
embedding = rep_net.predict([img, pose, embedding])

est_map = map_net.predict(embedding)[0]
frames.append((position, goal, est_map))

est_map = map_label

while len(frames) < MAX_STEPS:
    best_score = np.inf
    best_pos = position
    best_rot = 0
    for i in range(NUM_TRIES):
        rot = np.random.random() * 2 * np.pi
        new_pos = STEP_SIZE * np.array([np.cos(rot), np.sin(rot)]) + position
        min_coords = pos_to_coords(new_pos - ROBOT_RADIUS)
        max_coords = pos_to_coords(new_pos + ROBOT_RADIUS)
        prob = np.min(est_map[min_coords[0]:max_coords[0]+1, min_coords[1]:max_coords[1]+1])
        prob **= 4
        score = np.linalg.norm(goal - new_pos) * prob + MAX_DIST * (1 - prob)
        if score < best_score:
            best_score = score
            best_pos = new_pos
            best_rot = rot
    position = best_pos
    img, pose = get_image(position, best_rot)
    embedding = rep_net.predict([img, pose, embedding])
    est_map = map_net.predict(embedding)[0] > 0.5
    if np.linalg.norm(goal - position) < STEP_SIZE:
        goal = random_valid_pos(walls, map_label)
    frames.append((position, goal, est_map))

print(time.time() - start, 'for pathfinding')

img = np.zeros([MAP_SIZE, MAP_SIZE, 3])
img[:,:,2] = map_label

def animate(frame):
    pos, goal, est_map = frame
    plt.clf()
    img[:,:,0] = est_map
    img[:,:,1] = est_map * map_label
    plt.imshow(img, extent=(0, MAP_SIZE, MAP_SIZE, 0))
    plt.scatter(pos[1] * 8, pos[0] * 8, c='orange')
    plt.scatter(goal[1] * 8, goal[0] * 8, c='g')

plt.imshow(img, extent=(0, MAP_SIZE, MAP_SIZE, 0))
plt.show()
ani = FuncAnimation(plt.gcf(), animate, frames=frames, repeat=False)
plt.show()