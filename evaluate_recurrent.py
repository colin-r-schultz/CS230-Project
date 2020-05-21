import tensorflow as tf
import numpy as np
from constants import *
import os
import shutil

NUM_INPUT_OBS = 24
NUM_TEST_OBS = 1
EMBEDDING_SIZE = 512


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

class TileConvNet(tf.keras.layers.Layer):
    def __init__(self, tile_dims, output_dims, pretrained=False):
        super(TileConvNet, self).__init__()
        self.tile_dims = tile_dims
        weights = 'imagenet' if pretrained else None
        mobile_net = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights=weights)
        self.layers = mobile_net.layers
        self.intermediate_conv = tf.keras.layers.Conv2D(192, 1, 1, name='intermediate_conv')
        self.add_layers = ['block_1_project_BN', 'block_3_project_BN', 'block_4_add', 'block_6_project_BN', 
                           'block_7_add', 'block_8_add', 'block_10_project_BN', 'block_11_add', 'block_13_project_BN', 'block_14_add']
        self.average = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_dims)
    
    def call(self, inputs):
        image_input, pose_input = inputs
        pose_input = tf.tile(tf.reshape(pose_input, [-1, 1, 1, self.tile_dims]), [1, 16, 16, 1])
        x = image_input
        prev_block = None
        for layer in self.layers:
            if layer.name[-3:] == 'add':
                x = [x, prev_block]
            elif layer.name == 'block_5_expand_BN':
                x = tf.concat([pose_input, x], axis=3)
                x = self.intermediate_conv(x)
            x = layer(x)
            if layer.name in self.add_layers:
                prev_block = x
        x = self.average(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

class RepresentationNetwork(tf.keras.Model):
    def __init__(self, pretrained=False):
        super(RepresentationNetwork, self).__init__()
        self.state_size = EMBEDDING_SIZE
        self.output_size = EMBEDDING_SIZE
        self.encoder = TileConvNet(VIEW_DIM, 1024, pretrained=pretrained)
        self.relu = tf.keras.layers.ReLU()
        self.dense = tf.keras.layers.Dense(512, activation='relu')
        self.embedding_dense = tf.keras.layers.Dense(EMBEDDING_SIZE)
        self.weight_dense = tf.keras.layers.Dense(EMBEDDING_SIZE, activation='sigmoid')

    def build(self, input_shape):
        pass

    def call(self, inputs, states):
        image, pos = inputs
        embedding = states[0]
        x = self.encoder([image, pos])
        x = self.relu(x)
        x = tf.concat([x, embedding], axis=1)
        x = self.dense(x)
        embedding_out = self.embedding_dense(x)
        embedding_weight = self.weight_dense(x)
        x = embedding_out * (embedding_weight) + embedding * (1 - embedding_weight)
        return x, x

def representation_network(pretrained=False):
    img_input = tf.keras.Input([None] + IMG_SHAPE)
    pose_input = tf.keras.Input([None, VIEW_DIM])

    base = RepresentationNetwork(pretrained)
    rnn = tf.keras.layers.RNN(base)

    output = rnn(inputs=(img_input, pose_input), initial_state=None)
    model = tf.keras.Model([img_input, pose_input], output, name='representation_net')
    return model

class LocalizationNetwork(tf.keras.Model):
    def __init__(self, pretrained=False):
        super(LocalizationNetwork, self).__init__()
        self.encoder = TileConvNet(EMBEDDING_SIZE, VIEW_DIM, pretrained=pretrained)
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        image, embedding = inputs
        x = self.encoder([image, embedding])
        pos = x[:, :3]
        rot = x[:, 3:6]
        roll = x[:, 6:]
        mag = tf.norm(rot, axis=1)
        rot = tf.raw_ops.DivNoNan(rot, mag)
        x = tf.concat([pos, rot, roll], axis=1)
        return x


def mapping_network():
    generator = tf.keras.Sequential([
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(4096, activation='sigmoid'),
        tf.keras.layers.Reshape([MAP_SIZE, MAP_SIZE], name='reshape')
    ], name='mapping_net')
    return generator

representation_net = representation_network(True)
mapping_net = mapping_network()

representation_net.load_weights('checkpoints/recurrent_fc/repnet_{}.cpkt'.format(19))
mapping_net.load_weights('checkpoints/recurrent_fc/mapnet_{}.cpkt'.format(19))

os.mkdir('visuals')
dev_set = create_dataset('dev')
for f in dev_set.take(1):
    pass
inputs, labels, metadata = f
inp_obs, inp_vp, obs = inputs
embedding = representation_net([inp_obs, inp_vp])
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
np.savez('visuals/data.npz', perm=perm, path=strpath, input_vps=input_vps, map_estimate=map_estimate[0],
    map_label=get_single(map_label), label_vps=get_single(vp_labels))
