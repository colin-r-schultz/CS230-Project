import numpy as np
import pybullet as p
from constants import *
import scene_builder
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import tensorflow as tf
import time
import disect_model
import os
np.random.seed(123456)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

walls, map_label, bodies = scene_builder.build_scene_full()

STEP_SIZE = PIXEL_FRACTION
ROBOT_RADIUS = PIXEL_FRACTION / 2
NUM_TRIES = 64
MAX_DIST = SCENE_SIZE * np.sqrt(2)
MAX_STEPS = 200
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


def preprocess(img, view):
    return prepocess_image(img), normalize_view(view)

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
    return rgb, view

rep_net, map_net, loc_net = disect_model.build_model()

goal = random_valid_pos(walls, map_label)
position = random_valid_pos(walls, map_label)

start = time.time()

embedding = np.zeros([1, EMBEDDING_SIZE])
img, pose = get_image(position, np.random.random() * 2 * np.pi)
pimg, pose = preprocess(img, pose)
embedding = rep_net.predict([pimg, pose, embedding])

est_map = map_net.predict(embedding)[0]
frames.append((pose, pose, goal, est_map, img))

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
        prob = min((prob * 2) ** 2, 1)
        score = np.linalg.norm(goal - new_pos) * prob + MAX_DIST * (1 - prob)
        if score < best_score:
            best_score = score
            best_pos = new_pos
            best_rot = rot
    position = best_pos
    img, pose = get_image(position, best_rot)
    pimg, pose = preprocess(img, pose)
    est_pose = loc_net.predict([pimg, embedding])
    embedding = rep_net.predict([pimg, pose, embedding])
    est_map = map_net.predict(embedding)[0]
    if np.linalg.norm(goal - position) < STEP_SIZE:
        goal = random_valid_pos(walls, map_label)
    frames.append((pose, est_pose, goal, est_map, img))

print(time.time() - start, 'for pathfinding')

def plot_goal(ax, goal):
    ax.scatter(goal[1] * 8, goal[0] * 8, c='g')

def plot_pose(ax, pose, c):
    pose = pose[0]
    pos = (pose[:2] * 4 + 4) * 8
    rot = pose[3:5]
    rot = rot / np.linalg.norm(rot)
    ax.quiver(pos[1], pos[0], rot[1], rot[0], angles='xy')
    ax.scatter(pos[1], pos[0], c=c)


def animate(frame, axs):
    pose, est_pose, goal, est_map, img = frame
    for i in np.ndindex(axs.shape):
        axs[i].clear()
    axs[0, 0].matshow(map_label, extent=(0, MAP_SIZE, MAP_SIZE, 0))
    axs[1, 0].imshow(img, extent=(0, MAP_SIZE, MAP_SIZE, 0))
    axs[0, 1].imshow(est_map > 0.5, extent=(0, MAP_SIZE, MAP_SIZE, 0))
    axs[1, 1].imshow(est_map, extent=(0, MAP_SIZE, MAP_SIZE, 0))
    for i in [(0, 0), (0, 1), (1, 1)]:
        plot_goal(axs[i], goal)
        plot_pose(axs[i], est_pose, 'r')
        plot_pose(axs[i], pose, 'b')

fig, axs = plt.subplots(2, 2)
fig.set_size_inches(6.4, 6.4)
fig.tight_layout()
ani = FuncAnimation(fig, animate, frames=frames, fargs=(axs,), repeat=False)
writer = writers['ffmpeg'](fps=10, bitrate=1800)
ani.save('animation.mp4', writer=writer)