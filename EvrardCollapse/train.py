'''
    Single-GPU training.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import argparse
import time
from datetime import datetime
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1.compat.v1 as tf
import socket
import importlib
import os
import sys
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

baseDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.dirname(baseDir)
SPH3DDir = os.path.join(rootDir, 'SPH3D-GCN')

sys.path.append(baseDir)
sys.path.append(SPH3DDir)
sys.path.append(os.path.join(SPH3DDir, 'models'))
sys.path.append(os.path.join(SPH3DDir, 'utils'))
import data_util

curFeature = sys.argv[1]

class flags(object):
    def __init__(self):
        self.batch_size = 128
        self.max_epoch = 1000
        self.learning_rate = 1e-3
        self.gpu = 0
        self.momentum = 0.9
        self.optimizer = 'adam'
        self.decay_step = 1e6
        self.decay_rate = 0.7
        self.model = 'SPH3D_single_reg'
        self.config = 'Config'
        self.log_dir = 'log_evrard_4X'
        self.load_ckpt = None

FLAGS = flags()
BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(baseDir, FLAGS.model+'.py')
LOG_DIR = os.path.join(baseDir,FLAGS.log_dir, curFeature)
if not os.path.exists(LOG_DIR): os.system('mkdir -p %s' %LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (os.path.join(baseDir, 'train.py'), LOG_DIR)) # bkp of train procedure
os.system('cp %s.py %s' % (os.path.join(baseDir, FLAGS.config), LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')

HOSTNAME = socket.gethostname()

net_config = importlib.import_module(FLAGS.config)
# dataDir = os.path.join(baseDir, 'Outputs/ramdisk/%s_data_%s' % (net_config.mode, curFeature))
dataDir = os.path.join(baseDir, 'Outputs/%s_data_%s' % (net_config.mode, curFeature))
trainlist = [line.rstrip() for line in open(os.path.join(dataDir, 'train_files.txt'))]
testlist = [line.rstrip() for line in open(os.path.join(dataDir, 'test_files.txt'))]
NUM_POINT = net_config.num_input
NUM_FEATURES = net_config.num_cls
INPUT_DIM = 8

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.000001) # CLIP THE LEARNING RATE!
    return learning_rate


def placeholder_inputs(batch_size, num_point):
    input_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, INPUT_DIM))
    feature_pl = tf.placeholder(tf.int32, shape=(batch_size))

    return input_pl, feature_pl


def augment_fn(batch_input, batch_feature, augment_ratio=0.5):
    bsize, num_point, _ = batch_input.shape

    # shuffle the orders of samples in a batch
    idx = np.arange(bsize)
    np.random.shuffle(idx)
    batch_input = batch_input[idx,:,:]
    # batch_rho = batch_rho[idx,:]
    # batch_engy = batch_engy[idx,:]
    batch_feature = batch_feature[idx]

    # shuffle the point orders of each sample
    idx = np.arange(num_point)
    np.random.shuffle(idx)
    batch_input = batch_input[:,idx,:]
    # batch_rho = batch_rho[:,idx]
    # batch_engy = batch_engy[:,idx]
    # batch_xyz = data_util.shuffle_points(batch_xyz)

    # perform augmentation on the first np.int32(augment_ratio*bsize) samples
    augSize = np.int32(augment_ratio * bsize)
    augment_xyz = batch_input[0:augSize, :, 0:3]

    augment_xyz = data_util.rotate_point_cloud(augment_xyz)
    augment_xyz = data_util.rotate_perturbation_point_cloud(augment_xyz)
    augment_xyz = data_util.random_scale_point_cloud(augment_xyz)
    augment_xyz = data_util.shift_point_cloud(augment_xyz)

    batch_input[0:augSize, :, 0:3] = augment_xyz

    return batch_input, batch_feature


def parse_fn(item):
    features = tf.io.parse_single_example(
        item,
        features={
            'xyz_raw': tf.io.FixedLenFeature([], dtype=tf.string),
            'xyz_vel_raw': tf.io.FixedLenFeature([], dtype=tf.string),
            'rho_raw': tf.io.FixedLenFeature([], dtype=tf.string),
            'engy_raw': tf.io.FixedLenFeature([], dtype=tf.string),
            'feature': tf.io.FixedLenFeature([], dtype=tf.float32)})

    xyz = tf.io.decode_raw(features['xyz_raw'], tf.float32)
    vel = tf.io.decode_raw(features['xyz_vel_raw'], tf.float32)
    rho = tf.io.decode_raw(features['rho_raw'], tf.float32)
    engy = tf.io.decode_raw(features['engy_raw'], tf.float32)
    label = tf.cast(features['feature'], tf.float32)
    xyz = tf.reshape(xyz,[-1, 3])
    vel = tf.reshape(vel,[-1, 3])
    rho = tf.reshape(rho, [-1, 1])
    engy = tf.reshape(engy, [-1, 1])
    all_in_one = tf.concat((xyz, vel, rho, engy), axis=-1)
    return all_in_one, label


def input_fn(filelist, batch_size=16, buffer_size=10000):
    dataset = tf.data.TFRecordDataset(filelist)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.map(parse_fn, num_parallel_calls=4)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    # dataset = dataset.padded_batch(batch_size, padded_shapes=(None,INPUT_DIM+2), \
                                   # padding_values=-1.0, drop_remainder=False)

    return dataset

def train():
    # ===============================Prepare the Dataset===============================
    trainset = input_fn(trainlist, BATCH_SIZE, 10000)
    train_iterator = trainset.make_initializable_iterator()
    next_train_element = train_iterator.get_next()

    testset = input_fn(testlist, BATCH_SIZE, 10000)
    test_iterator = testset.make_initializable_iterator()
    next_test_element = test_iterator.get_next()
    # =====================================The End=====================================

    with tf.device('/gpu:0'):
        # =================================Define the Graph================================
        input_pl, feature_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)

        training_pl = tf.placeholder(tf.bool, shape=())
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get model and loss
        pred, end_points = MODEL.get_model(input_pl, training_pl, config=net_config)
        MODEL.get_loss(pred, feature_pl, end_points)
        if net_config.weight_decay is not None:
            reg_loss = tf.multiply(tf.losses.get_regularization_loss(), net_config.weight_decay, name='reg_loss')
            tf.add_to_collection('losses', reg_loss)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')
        tf.summary.scalar('total_loss', total_loss)
        for l in losses + [total_loss]:
            tf.summary.scalar(l.op.name, l)

        # correct = tf.equal(tf.argmax(pred, 1, output_type=tf.float32), tf.cast(feature_pl,tf.float32))
        # accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
        # tf.summary.scalar('accuracy', accuracy)


        print("--- Get training operator")
        # Get training operator
        learning_rate = get_learning_rate(global_step)
        tf.summary.scalar('learning_rate', learning_rate)
        if OPTIMIZER == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM, use_nesterov=True)
        elif OPTIMIZER == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step=global_step)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(max_to_keep=500)
        # =====================================The End=====================================

    n = len([n.name for n in tf.get_default_graph().as_graph_def().node])
    print("*****************The Graph has %d nodes*****************"%(n))

    # =================================Start a Session================================
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    with tf.Session(config=config) as sess:
        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        sess.run(init)  # Init variables

        # Load the model
        print(FLAGS.load_ckpt)
        if FLAGS.load_ckpt is not None:
            saver.restore(sess, FLAGS.load_ckpt)
            print('{}-Checkpoint loaded from {}!'.format(datetime.now(), FLAGS.load_ckpt))
        else:
            latest_ckpt = tf.train.latest_checkpoint(LOG_DIR)
            if latest_ckpt:
                print('{}-Found checkpoint {}'.format(datetime.now(), latest_ckpt))
                saver.restore(sess, latest_ckpt)
                print('{}-Checkpoint loaded from {} (Iter {})'.format(
                    datetime.now(), latest_ckpt, sess.run(global_step)))

        ops = {'input_pl': input_pl,
               'feature_pl': feature_pl,
               'training_pl': training_pl,
               'pred': pred,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'global_step': global_step,
               'end_points': end_points}

        if latest_ckpt:
            checkpoint_epoch = int(latest_ckpt.split('-')[-1])+1
        else:
            checkpoint_epoch = 0

        for epoch in range(checkpoint_epoch, MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            sess.run(train_iterator.initializer)
            train_one_epoch(sess, ops, next_train_element, train_writer)

            log_string(str(datetime.now()))
            log_string('---- EPOCH %03d EVALUATION ----' %(epoch))

            sess.run(test_iterator.initializer)
            eval_one_epoch(sess, ops, next_test_element, test_writer)

            save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step=epoch)
            log_string("Model saved in file: %s" % save_path)
    # =====================================The End=====================================


def train_one_epoch(sess, ops, next_train_element, train_writer):
    """ ops: dict mapping from string to tf ops """
    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_input = np.zeros((BATCH_SIZE, NUM_POINT, INPUT_DIM))
    cur_batch_feature = np.zeros((BATCH_SIZE), dtype=np.float32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0

    train_time = 0.0
    while True:
        try:
            batch_input, batch_feature = sess.run(next_train_element)
            bsize = batch_input.shape[0]

            batch_input[:, :, [0, 1, 2]] = batch_input[:, :, [0, 2, 1]]  # xzy to xyz
            batch_input, batch_feature = augment_fn(batch_input, batch_feature, augment_ratio=0.5)  # training augmentation on the fly
            # log_string('Augmentation done!')
            cur_batch_input[0:bsize,...] = batch_input
            # cur_batch_rho[0:bsize,:] = batch_rho
            # cur_batch_engy[0:bsize,:] = batch_engy
            cur_batch_feature[0:bsize] = batch_feature

            feed_dict = {ops['input_pl']: cur_batch_input,
                         ops['feature_pl']: cur_batch_feature,
                         ops['training_pl']: True}

            now = time.time()
            # log_string('To train ...')
            summary, global_step, _, loss_val, pred_val = sess.run([ops['merged'], ops['global_step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            train_time += (time.time() - now)
            # log_string('done!')

            train_writer.add_summary(summary, global_step)
            # pred_val = np.argmax(pred_val, 1)
            # correct = np.sum(pred_val[0:bsize] == batch_feature[0:bsize])
            # total_correct += correct
            # total_seen += bsize
            # loss_sum += loss_val
            loss_sum += loss_val/np.mean(batch_input[:,:,net_config.curFeatureIndex[curFeature]])
            # loss_sum += loss_val/np.mean(batch_input[:,:,net_config.curFeatureIndex[net_config.curFeature]])
            if (batch_idx+1)%50 == 0:
                log_string(' ---- batch: %03d ----' % (batch_idx+1))
                log_string('mean loss: %f' % (loss_sum / 50))
                # log_string('accuracy: %f' % (total_correct / float(total_seen)))
                # total_correct = 0
                # total_seen = 0
                loss_sum = 0
            batch_idx += 1
        except tf.errors.OutOfRangeError:
            break

    log_string("training one batch require %.2f milliseconds" %(1000*train_time/batch_idx))

    return


def eval_one_epoch(sess, ops, next_test_element, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    # Make sure batch data is of same size
    cur_batch_input = np.zeros((BATCH_SIZE, NUM_POINT, INPUT_DIM))
    cur_batch_feature = np.zeros((BATCH_SIZE), dtype=np.float32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    total_seen_class = [0 for _ in range(NUM_FEATURES)]
    total_correct_class = [0 for _ in range(NUM_FEATURES)]

    test_time = 0.0
    while True:
        try:
            batch_input, batch_feature = sess.run(next_test_element)
            bsize = batch_input.shape[0]

            batch_input[:, :, [0, 1, 2]] = batch_input[:, :, [0, 2, 1]]  # xzy to xyz
            cur_batch_input[0:bsize, ...] = batch_input
            cur_batch_feature[0:bsize] = batch_feature

            feed_dict = {ops['input_pl']:cur_batch_input,
                         ops['feature_pl']:cur_batch_feature,
                         ops['training_pl']:is_training}

            now = time.time()
            summary, global_step, loss_val, pred_val = sess.run([ops['merged'], ops['global_step'],
                                                          ops['loss'], ops['pred']], feed_dict=feed_dict)
            test_time += (time.time() - now)

            test_writer.add_summary(summary, global_step)
            # pred_val = np.argmax(pred_val, 1)
            # correct = np.sum(pred_val[0:bsize] == batch_feature[0:bsize])
            # total_correct += correct
            # total_seen += bsize
            # loss_sum += loss_val
            loss_sum += loss_val/np.mean(batch_input[:,:,net_config.curFeatureIndex[curFeature]])
            # loss_sum += loss_val/np.mean(batch_input[:,:,net_config.curFeatureIndex[net_config.curFeature]])
            batch_idx += 1
            # for i in range(0, bsize):
            #     l = batch_feature[i]
            #     total_seen_class[l] += 1
            #     total_correct_class[l] += (pred_val[i] == l)
        except tf.errors.OutOfRangeError:
            break

    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    # log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    # log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))

    log_string("testing one batch require %.2f milliseconds" % (1000 * test_time / batch_idx))

    return


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
