import tensorflow as tf
import dataloader
from weighted_loss import edge_weighted_loss
import sys
import architectures.recurrent_fc as model

dataloader.NUM_INPUT_OBS = model.NUM_INPUT_OBS
dataloader.NUM_TEST_OBS = model.NUM_TEST_OBS

BATCH_SIZE = 16
print('Creating datasets')
train_data = dataloader.create_dataset('datasets*', batch_size=BATCH_SIZE)
dev_data = dataloader.create_dataset('dev', batch_size=BATCH_SIZE)

print('Creating models')


representation_net = model.representation_network(True)
mapping_net = model.mapping_network()

load = 0
if len(sys.argv) > 1:
    load = int(sys.argv[1])
    representation_net.load_weights('checkpoints/{}/repnet_{}.cpkt'.format(model.CHECKPOINT_PATH, load))
    mapping_net.load_weights('checkpoints/{}/mapnet_{}.cpkt'.format(model.CHECKPOINT_PATH,load))


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

print('Training')
EPOCHS = 10
for epoch in range(load+1, EPOCHS+load+1):
    print('Starting epoch {}.'.format(epoch))

    for batch, ((inp_obs, inp_vp, test_obs), (vp_label, map_label))in enumerate(train_data):

        with tf.GradientTape() as tape:
            embedding = representation_net((inp_obs, inp_vp))
            map_estimate = mapping_net(embedding)
            loss, wloss = edge_weighted_loss(map_label, map_estimate)
        weights = representation_net.trainable_variables + mapping_net.trainable_variables
        grads = tape.gradient(wloss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        if batch % 200 == 0:
            print("Loss during batch {}: {}, {} (weighted)".format(batch, float(loss), float(wloss)))

    print('Saving Models')
    representation_net.save_weights('checkpoints/{}/repnet_{}.cpkt'.format(model.CHECKPOINT_PATH, epoch))
    mapping_net.save_weights('checkpoints/{}/mapnet_{}.cpkt'.format(model.CHECKPOINT_PATH, epoch))
print('Done!')