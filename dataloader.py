import numpy as np
import tensorflow as tf
from constants import *

NUM_INPUT_OBS = 16
NUM_TEST_OBS = 16

def open_npz(path):
    with np.load(path.numpy().decode() + '/labels.npz') as npz:
        map_label = npz['map_label']
        views = npz['views']
    perm = np.random.permutation(SHOTS_PER_SCENE)
    views = views[perm]
    assert (views.shape == (SHOTS_PER_SCENE, VIEW_DIM))
    return [map_label, views, perm]

def parse_scene(path):
    [map_label, views, perm] = tf.py_function(open_npz, [path], [tf.uint8, tf.float32, tf.int32])
    filenames = path + '/obs' + tf.as_string(perm) + '.png'
    image_list = []
    for i in range(SHOTS_PER_SCENE):
        decoded = tf.io.decode_png(tf.io.read_file(filenames[i]), channels=3)
        image = tf.image.convert_image_dtype(decoded, tf.float32)
        image = (image * 2) - 1
        image_list.append(image)
    images = tf.stack(image_list)

    views = (views - VIEW_MEANS) / VIEW_VARIATION
    map_label = tf.cast(map_label, tf.float32)
    return images[:NUM_INPUT_OBS], views[:NUM_INPUT_OBS], images[NUM_INPUT_OBS:NUM_INPUT_OBS+NUM_TEST_OBS], views[NUM_INPUT_OBS:NUM_INPUT_OBS+NUM_TEST_OBS], map_label

def ensure_shapes(inp_obs, inp_vp, obs, vp, map_label):
    inp_obs.set_shape([NUM_INPUT_OBS] + IMG_SHAPE)
    inp_vp.set_shape([NUM_INPUT_OBS, VIEW_DIM])
    map_label.set_shape([MAP_SIZE, MAP_SIZE])
    obs.set_shape([NUM_TEST_OBS] + IMG_SHAPE)
    vp.set_shape([NUM_TEST_OBS, VIEW_DIM])
    return inp_obs, inp_vp, obs, vp, map_label

def tuplify(inp_obs, inp_vp, obs, vp, map_label):
    return (inp_obs, inp_vp, obs), (vp, map_label)

def create_dataset(path, batch_size=16):
    list_scenes = tf.data.Dataset.list_files('data/{}/chunk*/scene*'.format(path)).shuffle(buffer_size=128)
    dataset = list_scenes.map(parse_scene).map(ensure_shapes).map(tuplify)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset