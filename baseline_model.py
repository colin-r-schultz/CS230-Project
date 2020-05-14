import tensorflow as tf
import numpy as np
from constants import *
import sys
import dataloader

NUM_INPUT_OBS = 16
NUM_TEST_OBS = 16
BATCH_SIZE = 16
EMBEDDING_SIZE = 256
dataloader.NUM_INPUT_OBS = NUM_INPUT_OBS
dataloader.NUM_TEST_OBS = NUM_TEST_OBS


class TileConvBlock(tf.keras.layers.Layer):
    def __init__(self, tile_dims, output_dims, **kwargs):
        super(TileConvBlock, self).__init__(**kwargs)
        self.tile_dims = tile_dims
        self.conv1 = tf.keras.layers.Conv2D(filters=256, kernel_size=2, strides=2, activation='relu', name='conv1')
        self.skip1 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, name='skip1')
        self.conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', name='conv2')
        self.conv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=2, strides=2, activation='relu', name='conv3')
        self.skip2 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, name='skip2')
        self.conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', name='conv4')
        self.conv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same', activation='relu', name='conv5')
        self.conv6 = tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, activation='relu', name='conv6')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu', name='dense1')
        self.dense2 = tf.keras.layers.Dense(output_dims, name='dense2')
    
    def call(self, inputs):
        image_input = inputs[0]
        pose_input = inputs[1]
        x = self.conv1(image_input)
        skip = self.skip1(x)
        x = self.conv2(x)
        x = x + skip
        x = self.conv3(x)
        pose_input = tf.tile(tf.reshape(pose_input, [-1, 1, 1, self.tile_dims]), [1, 32, 32, 1])
        x = tf.concat([pose_input, x], axis=3)
        skip = self.skip2(x)
        x = self.conv4(x)
        x = x + skip
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class RepresentationNetwork(tf.keras.Model):
    def __init__(self):
        super(RepresentationNetwork, self).__init__()
    
    def build(self, input_shape):
        self.encoder = TileConvBlock(VIEW_DIM, EMBEDDING_SIZE)

    def call(self, inputs):
        scene_input, views_input = inputs
        outs = []
        for i in range(NUM_INPUT_OBS):
            single_input = tf.reshape(scene_input[:,i], [-1] + IMG_SHAPE)
            views = tf.reshape(views_input[:, i], [-1, VIEW_DIM])
            outs.append(self.encoder([single_input, views]))
        embedding = tf.keras.layers.Add()(outs)
        return embedding

class LocalizationNetwork(tf.keras.Model):
    def __init__(self):
        super(LocalizationNetwork, self).__init__()

    def build(self, input_shape):
        self.encoder = TileConvBlock(EMBEDDING_SIZE, VIEW_DIM)
    
    def call(self, inputs):
        images_input, embedding = inputs
        outs = []
        for i in range(NUM_TEST_OBS):
            single_input = tf.reshape(images_input[:,i], [-1] + IMG_SHAPE)
            outs.append(tf.reshape(self.encoder([single_input, embedding]), [-1, 1, VIEW_DIM]))
        out = tf.concat(outs, axis=1)
        return out


def mapping_network():
    generator = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Reshape([16, 16, 16], name='reshape1'),
        tf.keras.layers.Conv2DTranspose(128, 4, 2, activation='relu', padding='same'),
        tf.keras.layers.Conv2DTranspose(128, 4, 2, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(1, 3, 1, activation='sigmoid', padding='same'),
        tf.keras.layers.Reshape([MAP_SIZE, MAP_SIZE], name='reshape2')
    ], name='mapping_net')
    return generator

tf.keras.backend.clear_session()
print('Creating datasets')
train = dataloader.create_dataset('datasets*')
dev = dataloader.create_dataset('dev')

print('Creating model')
if (len(sys.argv) > 1):
    e2e_model = tf.keras.models.load_model('checkpoints/'+sys.argv[1])
else:
    input_obs = tf.keras.Input(shape=[NUM_INPUT_OBS]+IMG_SHAPE, name='input_observations')
    input_poses = tf.keras.Input(shape=[NUM_INPUT_OBS, VIEW_DIM], name='input_poses')
    unknown_images = tf.keras.Input(shape=[NUM_TEST_OBS]+IMG_SHAPE, name='unknown_images')

    embedding = RepresentationNetwork()([input_obs, input_poses])
    pose_estimates = LocalizationNetwork()([unknown_images, embedding])
    map_estimate = mapping_network()(embedding)

    e2e_model = tf.keras.Model([input_obs, input_poses, unknown_images], [pose_estimates, map_estimate], name='end_to_end_model')
    e2e_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=['mse', 'binary_crossentropy'], loss_weights=[0.01, 1])

tb_callback = tf.keras.callbacks.TensorBoard(log_dir='tensorboard')
cp_callback = tf.keras.callbacks.ModelCheckpoint('checkpoints/baseline_model_{epoch}', verbose=1)

print('Training model')
e2e_model.fit(train, epochs=10, callbacks=[tb_callback, cp_callback], verbose=1)

print('Evaluating model')
e2e_model.evaluate(dev)


