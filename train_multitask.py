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

BATCH_SIZE = 16
if not TEST:
    train = dataloader.create_dataset('datasets*', batch_size=BATCH_SIZE)
dev = dataloader.create_dataset('dev', batch_size=BATCH_SIZE)

print('Creating models')
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

@tf.function
def map_loss_fn(y_true, y_pred):
    return edge_weighted_loss(y_true, y_pred, weight=32)[1]

@tf.function
def localization_loss_fn(y_true, y_pred):
    true_pos = y_true[:, :, :3]
    pred_pos = y_pred[:, :, :3]
    true_rot = y_true[:, :, 3:6]
    pred_rot = y_pred[:, :, 3:6]
    true_roll = y_true[:, :, 6]
    pred_roll = y_pred[:, :, 6]
    
    pos_loss = tf.keras.losses.mean_squared_error(true_pos, pred_pos)
    rot_loss = -tf.reduce_sum(true_rot * pred_rot, axis=-1, keepdims=False)
    roll_loss = tf.keras.losses.mean_squared_error(true_roll, pred_roll)

    return 2 * tf.reduce_mean(pos_loss) + tf.reduce_mean(rot_loss) + 0.5 * tf.reduce_mean(roll_loss)


load = None
if len(sys.argv) > 1:
    load = sys.argv[1]
    load = 'checkpoints/{}/e2e_model_{}.ckpt'.format(model.CHECKPOINT_PATH, load)

e2e_model = model.build_multitask_model(load)

e2e_model.compile(optimizer=optimizer, loss=[localization_loss_fn, map_loss_fn])

if not TEST:
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir='tensorboard/multitask')
    cp_callback = tf.keras.callbacks.ModelCheckpoint('checkpoints/'+model.CHECKPOINT_PATH+'/multi_model_best.ckpt', verbose=1, save_weights_only=True, save_best_only=True, monitor='val_loss')
    cp_callback_latest = tf.keras.callbacks.ModelCheckpoint('checkpoints/'+model.CHECKPOINT_PATH+'/multi_model_latest.ckpt', verbose=1, save_weights_only=True, save_best_only=False)
    cbs = [tb_callback, cp_callback, cp_callback_latest]


if TEST:
    print('Evaluating model')
    e2e_model.evaluate(dev)
    exit()

print('Training model')
e2e_model.fit(train, epochs=200, callbacks=cbs, verbose=2, validation_data=dev)
