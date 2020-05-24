import tensorflow as tf
import numpy as np
import architectures.recurrent_fc as model
import dataloader
from dataloader import ensure_shapes, tuplify, open_npz
from constants import *
import os
import sys
import shutil

NUM_INPUT_OBS = model.NUM_INPUT_OBS
NUM_TEST_OBS = model.NUM_TEST_OBS
dataloader.NUM_INPUT_OBS = model.NUM_INPUT_OBS
dataloader.NUM_TEST_OBS = model.NUM_TEST_OBS

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


def create_dataset(path):
    list_scenes = tf.data.Dataset.list_files('data/{}/chunk*/scene*'.format(path)).shuffle(buffer_size=128)
    dataset = list_scenes.map(parse_scene).map(ensure_shapes).map(tuplify)
    dataset = dataset.batch(1)

    return dataset

def get_single(tensor):
    return tensor.numpy()[0]

representation_net = model.representation_network(True)
mapping_net = model.mapping_network()

if len(sys.argv) > 1:
    load = int(sys.argv[1])
    representation_net.load_weights('checkpoints/{}/repnet_{}.cpkt'.format(model.CHECKPOINT_PATH, load))
    mapping_net.load_weights('checkpoints/{}/mapnet_{}.cpkt'.format(model.CHECKPOINT_PATH,load))

os.mkdir('visuals')
dev_set = create_dataset('dev')
for f in dev_set.take(1):
    pass
inputs, labels, metadata = f
inp_obs, inp_vp, test_obs = inputs
embedding = representation_net((inp_obs, inp_vp))
map_estimate = mapping_net(embedding)
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
    est_vps=0, map_estimate=get_single(map_estimate),
    map_label=get_single(map_label), label_vps=get_single(vp_labels))