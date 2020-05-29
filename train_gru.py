import tensorflow as tf
import numpy as np
from constants import *
import sys
import dataloader
import architectures.recurrent_gru as model
from weighted_loss import edge_weighted_loss

dataloader.NUM_INPUT_OBS = model.NUM_INPUT_OBS
dataloader.NUM_TEST_OBS = model.NUM_TEST_OBS

TEST = len(sys.argv) > 2 and sys.argv[2] == 'test'

print('Creating datasets')
def get_relevant(inp, label):
    inp_obs, inp_vp, obs = inp
    vp, map_label = label
    return (inp_obs, inp_vp), map_label

BATCH_SIZE = 16
if not TEST:
    train = dataloader.create_dataset('datasets*', batch_size=BATCH_SIZE).map(get_relevant)
dev = dataloader.create_dataset('dev', batch_size=BATCH_SIZE).map(get_relevant)

print('Creating models')
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

@tf.function
def loss_fn(y_true, y_pred):
    return edge_weighted_loss(y_true, y_pred, weight=32)[1]

e2e_model = model.build_e2e_model()

if len(sys.argv) > 1:
    load = int(sys.argv[1])
    e2e_model.load_weights('checkpoints/{}/e2e_model_{}'.format(model.CHECKPOINT_PATH, load))

e2e_model.compile(optimizer=optimizer, loss=loss_fn)

if not TEST:
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir='tensorboard')
    cp_callback = tf.keras.callbacks.ModelCheckpoint('checkpoints/'+model.CHECKPOINT_PATH+'/e2e_model_{epoch}.ckpt', verbose=1, save_weights_only=True)


if TEST:
    print('Evaluating model')
    e2e_model.evaluate(dev)
    exit()

print('Training model')
e2e_model.fit(train, epochs=200, callbacks=[tb_callback, cp_callback], verbose=2, validation_data=dev)
