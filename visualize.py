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
map_estimate = data['map_estimate']

y_true = tf.reshape(tf.convert_to_tensor(actual_map), [-1, MAP_SIZE, MAP_SIZE, 1])
loss_filter = tf.ones([3, 3, 1, 1])
conv = tf.nn.convolution(y_true, loss_filter, padding='SAME')
reshaped = tf.reshape(conv, [MAP_SIZE, MAP_SIZE])
mask = tf.cast(tf.math.logical_and(reshaped < 8.5, reshaped > 0.5), dtype=tf.float32)

label_views = data['label_vps']
perm = data['perm']
plt.matshow(mask.numpy(), extent=(0, MAP_SIZE, MAP_SIZE, 0))

# plot_points(*process_vps(given_views), 'g', offset=0)
# plot_points(*process_vps(label_views), 'r', offset=NUM_INPUT_OBS)
    

plt.show()