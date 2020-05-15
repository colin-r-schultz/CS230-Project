import numpy as np
import tensorflow as tf
import dataloader
from constants import *
import random

NUM_INPUT_OBS = 16
dataloader.NUM_INPUT_OBS = NUM_INPUT_OBS
dataloader.NUM_TEST_OBS = 1

EMBEDDING_SIZE = 256

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
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Reshape([16, 16, 16], name='reshape1'),
        tf.keras.layers.Conv2DTranspose(128, 4, 2, activation='relu', padding='same'),
        tf.keras.layers.Conv2DTranspose(128, 4, 2, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(1, 3, 1, activation='sigmoid', padding='same'),
        tf.keras.layers.Reshape([MAP_SIZE, MAP_SIZE], name='reshape2')
    ], name='mapping_net')
    return generator

def get_relevant(inputs, labels):
    inp_obs, inp_vp, obs = inputs
    vp, map_label = labels
    return (inp_obs, inp_vp), map_label

BATCH_SIZE = 16
print('Creating datasets')
train_data = dataloader.create_dataset('datasets*', batch_size=BATCH_SIZE).map(get_relevant)
dev_data = dataloader.create_dataset('dev', batch_size=BATCH_SIZE).map(get_relevant)

print('Creating models')

fake_data = [tf.random.uniform([BATCH_SIZE, NUM_INPUT_OBS] + IMG_SHAPE, minval=-1, maxval=1), 
tf.random.uniform([BATCH_SIZE, NUM_INPUT_OBS, VIEW_DIM], minval=-1, maxval=1)]

# img_input = tf.keras.Input([None] + IMG_SHAPE)
# pose_input = tf.keras.Input([None, VIEW_DIM])
representation_net = representation_network(True)
mapping_net = mapping_network()

# embedding = representation_net([img_input, pose_input])
# map_estimate = mapping_net(embedding)

# e2e_model = tf.keras.Model([img_input, pose_input], map_estimate)
# print(e2e_model(fake_data).numpy())

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# e2e_model.compile(optimizer=optimizer, loss='binary_crossentropy')

print('Training')
# tb_callback = tf.keras.callbacks.TensorBoard(log_dir='tensorboard')
# cp_callback = tf.keras.callbacks.ModelCheckpoint('checkpoints/recurrent_model_{epoch}', verbose=1, save_weights_only=True)
EPOCHS = 10
# e2e_model.fit(train_data, epochs=EPOCHS, verbose=2, callbacks=[tb_callback, cp_callback])
for epoch in range(EPOCHS):
    print('Starting epoch {}.'.format(epoch))

    for batch, (inputs, map_label)in enumerate(train_data):
        # sequence_length = int(random.random() * 8 + 9)
        # inp_obs = inp_obs[:,:sequence_length]
        # inp_vp = inp_vp[:,:sequence_length]
        with tf.GradientTape() as tape:
            embedding = representation_net(inputs)
            map_estimate = mapping_net(embedding)# e2e_model([inp_obs, inp_vp])
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(map_label, map_estimate))
        weights = representation_net.trainable_variables + mapping_net.trainable_variables
        grads = tape.gradient(loss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        if batch % 200 == 0:
            print("Loss during batch {}: {}".format(batch, float(loss)))

    print('Saving Models')
    representation_net.save_weights('checkpoints/recurrent/repnet_{}.cpkt'.format(epoch))
    mapping_net.save_weights('checkpoints/recurrent/repnet_{}.cpkt'.format(epoch))
# print('Evaluating model')
# e2e_model.evaluate(dev_data, verbose=1)
print('Done!')
