import tensorflow as tf
from constants import *
import architectures.recurrent_gru as model

EMBEDDING_SIZE = model.EMBEDDING_SIZE

"""
This module is used to build the multitask model in its original configuration in order to load the weights,
and then reconfiguring it into 3 separate models.
"""

def representation_network(pretrained=False):
    img_input = tf.keras.Input([None] + IMG_SHAPE)
    pose_input = tf.keras.Input([None, VIEW_DIM])

    seq_length = 16

    reshaped_img_input = tf.reshape(img_input, [-1] + IMG_SHAPE)
    reshaped_pose_input = tf.reshape(pose_input, [-1, VIEW_DIM])

    conv = model.TileConvNet(VIEW_DIM, EMBEDDING_SIZE, pretrained)

    features = conv([reshaped_img_input, reshaped_pose_input])

    features = tf.reshape(features, [-1, seq_length, EMBEDDING_SIZE])
    gru = tf.keras.layers.GRU(EMBEDDING_SIZE)
    embedding = gru(features)

    repmodel = tf.keras.Model([img_input, pose_input], embedding, name='representation_net')
    return repmodel, conv, gru

def localization_network(pretrained=False):
    img_input = tf.keras.Input([None] + IMG_SHAPE)
    embedding = tf.keras.Input([EMBEDDING_SIZE])

    seq_length = model.NUM_TEST_OBS

    embedding_repeat = tf.keras.layers.RepeatVector(seq_length)(embedding)

    reshaped_img_input = tf.reshape(img_input, [-1] + IMG_SHAPE)
    reshaped_embedding = tf.reshape(embedding_repeat, [-1, EMBEDDING_SIZE])

    conv = model.TileConvNet(EMBEDDING_SIZE, VIEW_DIM, pretrained)
    x = conv([reshaped_img_input, reshaped_embedding])
    x = model.normalize_output(x)

    x = tf.reshape(x, [-1, seq_length, VIEW_DIM])

    locmodel = tf.keras.Model([img_input, embedding], x, name='localization_net')
    return locmodel, conv

def build_model():

    img_input = tf.keras.Input([None] + IMG_SHAPE)
    pose_input = tf.keras.Input([None, VIEW_DIM])
    unknown_img_input = tf.keras.Input([None] + IMG_SHAPE)

    rep_net, conv, gru = representation_network() 
    embedding = rep_net([img_input, pose_input])
    mapping_net = model.mapping_network()
    map_estimate = mapping_net(embedding)
    loc_net, conv2 = localization_network()
    est_vps = loc_net([unknown_img_input, embedding])



    e2e_model = tf.keras.Model([img_input, pose_input, unknown_img_input], [est_vps, map_estimate], name='e2e_model')
    e2e_model.load_weights('checkpoints/' + model.CHECKPOINT_PATH + '/multi_model_best.ckpt')

    img_input2 = tf.keras.Input(IMG_SHAPE)
    pose_input2 = tf.keras.Input([VIEW_DIM])
    features = conv([img_input2, pose_input2])
    prev_embedding = tf.keras.Input([EMBEDDING_SIZE])
    embedding = gru(tf.reshape(features, [-1, 1, EMBEDDING_SIZE]), initial_state=prev_embedding)
    rep_net2 = tf.keras.Model([img_input2, pose_input2, prev_embedding], embedding, name='rep_net')
    unknown_img_input2 = tf.keras.Input(IMG_SHAPE)
    embedding_in = tf.keras.Input([EMBEDDING_SIZE])
    est_vp = model.normalize_output(conv2([unknown_img_input2, embedding_in]))
    loc_net2 = tf.keras.Model([unknown_img_input2, embedding_in], est_vp, name='localization_net')
    return rep_net2, mapping_net, loc_net2