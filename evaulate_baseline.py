import tensorflow as tf
import numpy as np
from constants import *
import os
import shutil

NUM_INPUT_OBS = 16
NUM_TEST_OBS = 16
BATCH_SIZE = 16
IMG_SHAPE = [IMAGE_HEIGHT, IMAGE_WIDTH, 3]
EMBEDDING_SIZE = 256
VIEW_MEANS = [SCENE_SIZE / 2, SCENE_SIZE / 2, PUPPER_HEIGHT - HEIGHT_VARIATION / 2, 0, 0, 0, 0]
VIEW_VARIATION = [SCENE_SIZE / 2, SCENE_SIZE / 2, HEIGHT_VARIATION / 2, 1, 1, 1, ROLL_VARIATION]


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
    return images[:NUM_INPUT_OBS], views[:NUM_INPUT_OBS], images[NUM_INPUT_OBS:NUM_INPUT_OBS+NUM_TEST_OBS], views[NUM_INPUT_OBS:NUM_INPUT_OBS+NUM_TEST_OBS], map_label, perm, path

def ensure_shapes(inp_obs, inp_vp, obs, vp, map_label, perm, path):
    inp_obs.set_shape([NUM_INPUT_OBS] + IMG_SHAPE)
    inp_vp.set_shape([NUM_INPUT_OBS, VIEW_DIM])
    map_label.set_shape([MAP_SIZE, MAP_SIZE])
    obs.set_shape([NUM_TEST_OBS] + IMG_SHAPE)
    vp.set_shape([NUM_TEST_OBS, VIEW_DIM])
    perm.set_shape([SHOTS_PER_SCENE])
    return inp_obs, inp_vp, obs, vp, map_label, perm, path

def tuplify(inp_obs, inp_vp, obs, vp, map_label, perm, path):
    return (inp_obs, inp_vp, obs), (vp, map_label), (perm, path)

def create_dataset(path):
    list_scenes = tf.data.Dataset.list_files('data/{}/chunk*/scene*'.format(path)).shuffle(buffer_size=128)
    dataset = list_scenes.map(parse_scene).map(ensure_shapes).map(tuplify)
    dataset = dataset.batch(1)

    return dataset

def get_single(tensor):
    return tensor.numpy()[0]

os.mkdir('visuals')
e2e_model = tf.keras.models.load_model('checkpoints/baseline_model_1')
e2e_model.load_weights('checkpoints/baseline_model_06')
dev_set = create_dataset('dev')
for f in dev_set.take(1):
    pass
inputs, labels, metadata = f
vp_estimates, map_estimates = e2e_model.predict(inputs)
perm, path = metadata
vp_labels, map_label = labels
input_vps = get_single(inputs[1])


strpath = get_single(path).decode()
perm = get_single(perm)

def move_images_to_folder(path, folder, array):
    os.mkdir(folder)
    for i in range(len(array)):
        fname = '/obs{}.png'.format(array[i])
        src = path + fname
        dst = folder + fname
        shutil.copyfile(src, dst)

move_images_to_folder(strpath, 'visuals/given_images', perm[:NUM_INPUT_OBS])
move_images_to_folder(strpath, 'visuals/unknown_images', perm[NUM_INPUT_OBS:NUM_INPUT_OBS+NUM_TEST_OBS])
np.savez('visuals/data.npz', perm=perm, path=strpath, input_vps=input_vps, 
    est_vps=get_single(vp_estimates), map_estimate=get_single(map_estimates),
    map_label=get_single(map_label), label_vps=get_single(vp_labels))
