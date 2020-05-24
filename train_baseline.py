import tensorflow as tf
import numpy as np
from constants import *
import sys
import dataloader
import architectures.baseline as baseline

dataloader.NUM_INPUT_OBS = baseline.NUM_INPUT_OBS
dataloader.NUM_TEST_OBS = baseline.NUM_TEST_OBS

train = dataloader.create_dataset('datasets*')
dev = dataloader.create_dataset('dev')

e2e_model = baseline.build_e2e_model()

tb_callback = tf.keras.callbacks.TensorBoard(log_dir='tensorboard')
cp_callback = tf.keras.callbacks.ModelCheckpoint('checkpoints/baseline_model_{epoch}', verbose=1)

print('Training model')
e2e_model.fit(train, epochs=10, callbacks=[tb_callback, cp_callback], verbose=1)

print('Evaluating model')
e2e_model.evaluate(dev)


