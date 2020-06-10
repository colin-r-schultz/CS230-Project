import numpy as np
import matplotlib.pyplot as plt
from constants import *
import matplotlib.collections as mc
# import tensorflow as tf

NUM_INPUT_OBS = 16
NUM_TEST_OBS = 8

LABEL_POINTS = False

data = np.load('visuals/data.npz')

def process_vps(vps):
    pos = (vps[:,:2] * 4 + 4) * 8
    rot = vps[:, 3:5]
    rot = rot / np.linalg.norm(rot, axis=1, keepdims=True)
    return pos.T, rot.T, 

def plot_points(pos, rot, c, offset=0, ax=plt):
    ax.quiver(pos[1], pos[0], rot[1], rot[0], angles='xy')
    ax.scatter(pos[1], pos[0], c=c)
    if LABEL_POINTS:
        for i in range(pos.shape[1]):
            ax.annotate(perm[offset:][i], xy=(pos[1, i], pos[0, i]), c='w')

def plot_doubles(pts1, pts2, offset=0, ax=plt):
    pos1, rot1 = process_vps(pts1)
    pos2, rot2 = process_vps(pts2)
    plot_points(pos1, rot1, c='g', offset=offset, ax=ax)
    plot_points(pos2, rot2, c='r', offset=offset, ax=ax)
    lines, colors = [], []
    for i in range(pts1.shape[0]):
        lines.append([(pos1[1, i], pos1[0, i]), (pos2[1, i], pos2[0, i])])
    lc = mc.LineCollection(lines, colors='r', linewidths=1, linestyles='solid')
    ax.add_collection(lc)
    


given_views = data['input_vps']
actual_map = data['map_label']
map_estimate = data['map_estimate']

label_views = data['label_vps']
est_views = data['est_vps']
perm = data['perm']

fig, axs = plt.subplots(1, 3)
axs[0].matshow(actual_map, extent=(0, MAP_SIZE, MAP_SIZE, 0))
plot_points(*process_vps(given_views), 'g', offset=0, ax=axs[0])

axs[1].matshow(map_estimate, extent=(0, MAP_SIZE, MAP_SIZE, 0))

axs[2].matshow(map_estimate > 0.5, extent=(0, MAP_SIZE, MAP_SIZE, 0))
plot_doubles(label_views, est_views, ax=axs[2])

print(fig.get_size_inches())
fig.set_size_inches(4.8 * 3, 4.8)
fig.tight_layout()
plt.show()