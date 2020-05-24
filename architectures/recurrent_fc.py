import tensorflow as tf
import architectures.recurrent_deconv as recurrent_deconv
from architectures.recurrent_deconv import representation_network
from constants import *

NUM_INPUT_OBS = 24
NUM_TEST_OBS = 1 # doesn't use test obs (no localization)

EMBEDDING_SIZE = 512
recurrent_deconv.EMBEDDING_SIZE = EMBEDDING_SIZE

CHECKPOINT_PATH = 'recurrent_fc_v2'

def mapping_network():
    generator = tf.keras.Sequential([
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(4096, activation='sigmoid'),
        tf.keras.layers.Reshape([MAP_SIZE, MAP_SIZE], name='reshape')
    ], name='mapping_net')
    return generator