import tensorflow as tf
from constants import *

loss_filter = tf.ones([3, 3, 1, 1])

def edge_weighted_loss(y_true, y_pred, weight=64):
    y_true = tf.reshape(y_true, [-1, MAP_SIZE, MAP_SIZE, 1])
    y_pred = tf.reshape(y_pred, [-1, MAP_SIZE, MAP_SIZE, 1])
    unweighted_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    conv = tf.nn.convolution(y_true, loss_filter, padding='SAME')
    reshaped = tf.reshape(conv, [-1, MAP_SIZE, MAP_SIZE])
    mask = tf.cast(tf.math.logical_and(reshaped < 8.5, reshaped > 0.5), dtype=tf.float32)
    weighted_mask = mask * (weight - 1) + 1
    weighted_loss = weighted_mask * unweighted_loss
    return tf.reduce_mean(unweighted_loss), tf.reduce_mean(weighted_loss)