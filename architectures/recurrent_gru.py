import tensorflow as tf
from constants import *

NUM_INPUT_OBS = 16
NUM_TEST_OBS = 16 # doesn't use test obs (no localization)

EMBEDDING_SIZE = 512

CHECKPOINT_PATH = 'recurrent_gru'

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
        self.dense1 = tf.keras.layers.Dense(output_dims)
    
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
        return x

def representation_network(pretrained=False):
    img_input = tf.keras.Input([None] + IMG_SHAPE)
    pose_input = tf.keras.Input([None, VIEW_DIM])

    seq_length = NUM_INPUT_OBS

    reshaped_img_input = tf.reshape(img_input, [-1] + IMG_SHAPE)
    reshaped_pose_input = tf.reshape(pose_input, [-1, VIEW_DIM])

    features = TileConvNet(VIEW_DIM, EMBEDDING_SIZE, pretrained)([reshaped_img_input, reshaped_pose_input])

    features = tf.reshape(features, [-1, seq_length, EMBEDDING_SIZE])
    embedding = tf.keras.layers.GRU(EMBEDDING_SIZE)(features)

    model = tf.keras.Model([img_input, pose_input], embedding, name='representation_net')
    return model

def localization_network(pretrained=False):
    img_input = tf.keras.Input([None] + IMG_SHAPE)
    embedding = tf.keras.Input([EMBEDDING_SIZE])

    seq_length = NUM_TEST_OBS

    embedding_repeat = tf.keras.layers.RepeatVector(seq_length)(embedding)

    reshaped_img_input = tf.reshape(img_input, [-1] + IMG_SHAPE)
    reshaped_embedding = tf.reshape(embedding_repeat, [-1, EMBEDDING_SIZE])

    x = TileConvNet(EMBEDDING_SIZE, VIEW_DIM, pretrained)([reshaped_img_input, reshaped_embedding])
    pos = x[:, :3]
    rot = x[:, 3:6]
    roll = x[:, 6:]
    mag = tf.norm(rot, axis=1, keepdims=True)
    rot = tf.raw_ops.DivNoNan(x=rot, y=mag)
    x = tf.concat([pos, rot, roll], axis=1)

    x = tf.reshape(x, [-1, seq_length, VIEW_DIM])
    model = tf.keras.Model([img_input, embedding], x, name='localization_net')
    return model


def mapping_network():
    generator = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dense(4096, activation='sigmoid'),
        tf.keras.layers.Reshape([MAP_SIZE, MAP_SIZE], name='reshape')
    ], name='mapping_net')
    return generator

def build_e2e_model():
    img_input = tf.keras.Input([None] + IMG_SHAPE)
    pose_input = tf.keras.Input([None, VIEW_DIM])

    embedding = representation_network(True)([img_input, pose_input])
    map_estimate = mapping_network()(embedding)

    return tf.keras.Model([img_input, pose_input], map_estimate, name='e2e_model')

def build_multitask_model():
    img_input = tf.keras.Input([None] + IMG_SHAPE)
    pose_input = tf.keras.Input([None, VIEW_DIM])
    unknown_img_input = tf.keras.Input([None] + IMG_SHAPE)

    embedding = representation_network(True)([img_input, pose_input])
    est_vps = localization_network(True)([unknown_img_input, embedding])
    map_estimate = mapping_network()(embedding)

    return tf.keras.Model([img_input, pose_input, unknown_img_input], [est_vps, map_estimate], name='multitask_model')