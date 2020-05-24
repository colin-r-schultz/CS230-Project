import numpy as np
import matplotlib.pyplot as plt
from constants import *
import tensorflow as tf

NUM_INPUT_OBS = 24
NUM_TEST_OBS = 1

LABEL_POINTS = True

data = np.load('visuals/data.npz')

def process_vps(vps):
    pos = (vps[:,:2] * 4 + 4) * 8
    rot = vps[:, 3:5]
    rot = rot / np.linalg.norm(rot, axis=1, keepdims=True)
    return pos.T, rot.T, 

def plot_points(pos, rot, c, offset=0):
    plt.quiver(pos[1], pos[0], rot[1], rot[0], angles='xy')
    plt.scatter(pos[1], pos[0], c=c)
    if LABEL_POINTS:
        for i in range(NUM_INPUT_OBS):
            plt.annotate(perm[offset:][i], xy=(pos[1, i], pos[0, i]), c='w')

given_views = data['input_vps']
actual_map = data['map_label']

label_views = data['label_vps']
perm = data['perm']
plt.matshow(actual_map, extent=(0, MAP_SIZE, MAP_SIZE, 0))

plot_points(*process_vps(given_views), 'g', offset=0)
# plot_points(*process_vps(label_views), 'r', offset=NUM_INPUT_OBS)
    

plt.show()