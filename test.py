import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import pc_util


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        # loss = MODEL.get_loss(pred, labels_pl, end_points)
        loss, _ = MODEL.get_loss(pred, pointclouds_pl, end_points)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss,
           'pt_2d': end_points['pts_2d']}

    eval_one_epoch(sess, ops, num_votes)

   
def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
    for fn in range(len(TEST_FILES)):
        log_string('----'+str(fn)+'----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        print(current_data.shape)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        print(file_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            cur_batch_size = end_idx - start_idx
            
            # Aggregating BEG
            batch_loss_sum = 0 # sum of losses for the batch
            batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES)) # score for classes
            batch_pred_classes = np.zeros((cur_batch_size, NUM_CLASSES)) # 0/1 for classes
            for vote_idx in range(num_votes):
                rotated_data = provider.rotate_point_cloud_by_angle(current_data[start_idx:end_idx, :, :],
                                                  vote_idx/float(num_votes) * np.pi * 2)
                feed_dict = {ops['pointclouds_pl']: rotated_data,
                             ops['labels_pl']: current_label[start_idx:end_idx],
                             ops['is_training_pl']: is_training}
                loss_val, pred_val, pt_2d_val = sess.run([ops['loss'], ops['pred'], ops['pt_2d']],
                                          feed_dict=feed_dict)
                
                
                
                
                
                
                
                # test, kaidong
                import pdb
                # pdb.set_trace()
                import matplotlib.pyplot as plt
                pt_2d_val = np.expand_dims(pt_2d_val, 2)
                fig = plt.figure()
                ax = fig.add_subplot()
                ax.scatter(pt_2d_val[0, :, 0, 0], pt_2d_val[0, :, 0, 1], s=2 )
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                plt.savefig(os.path.join(DUMP_DIR, '2d_pt_fully_no_trans_{}_{}.png'.format( SHAPE_NAMES[current_label[start_idx]], start_idx )))
                

                # from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(rotated_data[0, :, 0], rotated_data[0, :, 1], rotated_data[0, :, 2], s=5)
                ax.view_init(85, 45)
                plt.savefig(os.path.join(DUMP_DIR, '3d_pt_{}_{}.png'.format( SHAPE_NAMES[current_label[start_idx]], start_idx )))
                

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(pred_val[0, :, 0], pred_val[0, :, 1], pred_val[0, :, 2], s=5)
                ax.view_init(85, 45)
                plt.savefig(os.path.join(DUMP_DIR, '3d_pt_{}_{}_recon.png'.format( SHAPE_NAMES[current_label[start_idx]], start_idx )))
                pdb.set_trace()
                
                
                
                
                # batch_pred_sum += pred_val
                # batch_pred_val = np.argmax(pred_val, 1)
                # for el_idx in range(cur_batch_size):
                #     batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
                batch_loss_sum += (loss_val * cur_batch_size / float(num_votes))
            # # pred_val_topk = np.argsort(batch_pred_sum, axis=-1)[:,-1*np.array(range(topk))-1]
            # # pred_val = np.argmax(batch_pred_classes, 1)
            # pred_val = np.argmax(batch_pred_sum, 1)
            # # Aggregating END
            
            # correct = np.sum(pred_val == current_label[start_idx:end_idx])
            # # correct = np.sum(pred_val_topk[:,0:topk] == label_val)
            # total_correct += correct
            total_seen += cur_batch_size
            loss_sum += batch_loss_sum

            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                # total_correct_class[l] += (pred_val[i-start_idx] == l)
                # fout.write('%d, %d\n' % (pred_val[i-start_idx], l))
                
                # if pred_val[i-start_idx] != l and FLAGS.visu: # ERROR CASE, DUMP!
                #     img_filename = '%d_label_%s_pred_%s.jpg' % (error_cnt, SHAPE_NAMES[l],
                #                                            SHAPE_NAMES[pred_val[i-start_idx]])
                #     img_filename = os.path.join(DUMP_DIR, img_filename)
                #     output_img = pc_util.point_cloud_three_views(np.squeeze(current_data[i, :, :]))
                #     scipy.misc.imsave(img_filename, output_img)
                #     error_cnt += 1
                
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    # log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    # log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    
    # class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
    # for i, name in enumerate(SHAPE_NAMES):
    #     log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))
    


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=1)
    LOG_FOUT.close()




















# import argparse
# import math
# import h5py
# import numpy as np
# import tensorflow as tf
# import socket
# import importlib
# import os
# import sys
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(BASE_DIR, 'models'))
# sys.path.append(os.path.join(BASE_DIR, 'utils'))
# import provider
# import tf_util

# from loss import tf_nndistance
# import copy

# parser = argparse.ArgumentParser()
# parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
# parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
# parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
# parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
# parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
# parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
# parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
# parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
# parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
# parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
# parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
# FLAGS = parser.parse_args()


# BATCH_SIZE = FLAGS.batch_size
# NUM_POINT = FLAGS.num_point
# MAX_EPOCH = FLAGS.max_epoch
# BASE_LEARNING_RATE = FLAGS.learning_rate
# GPU_INDEX = FLAGS.gpu
# MOMENTUM = FLAGS.momentum
# OPTIMIZER = FLAGS.optimizer
# DECAY_STEP = FLAGS.decay_step
# DECAY_RATE = FLAGS.decay_rate

# MODEL = importlib.import_module(FLAGS.model) # import network module
# MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
# LOG_DIR = FLAGS.log_dir
# if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
# os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
# os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
# LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
# LOG_FOUT.write(str(FLAGS)+'\n')

# MAX_NUM_POINT = 2048
# NUM_CLASSES = 40

# BN_INIT_DECAY = 0.5
# BN_DECAY_DECAY_RATE = 0.5
# BN_DECAY_DECAY_STEP = float(DECAY_STEP)
# BN_DECAY_CLIP = 0.99

# HOSTNAME = socket.gethostname()

# # ModelNet40 official train/test split
# TRAIN_FILES = provider.getDataFiles( \
#     os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
# TEST_FILES = provider.getDataFiles(\
#     os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

# def log_string(out_str):
#     LOG_FOUT.write(out_str+'\n')
#     LOG_FOUT.flush()
#     print(out_str)


# def get_learning_rate(batch):
#     learning_rate = tf.train.exponential_decay(
#                         BASE_LEARNING_RATE,  # Base learning rate.
#                         batch * BATCH_SIZE,  # Current index into the dataset.
#                         DECAY_STEP,          # Decay step.
#                         DECAY_RATE,          # Decay rate.
#                         staircase=True)
#     learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
#     return learning_rate        

# def get_bn_decay(batch):
#     bn_momentum = tf.train.exponential_decay(
#                       BN_INIT_DECAY,
#                       batch*BATCH_SIZE,
#                       BN_DECAY_DECAY_STEP,
#                       BN_DECAY_DECAY_RATE,
#                       staircase=True)
#     bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
#     return bn_decay

# def train():
#     with tf.Graph().as_default():
#         with tf.device('/gpu:'+str(GPU_INDEX)):
#             pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
#             is_training_pl = tf.placeholder(tf.bool, shape=())
#             print(is_training_pl)
            
#             # Note the global_step=batch parameter to minimize. 
#             # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
#             batch = tf.Variable(0)
#             bn_decay = get_bn_decay(batch)
#             tf.summary.scalar('bn_decay', bn_decay)

#             # Get model and loss 
#             pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
#             # loss, _ = MODEL.get_loss(pred, labels_pl, end_points)
#             loss, _ = MODEL.get_loss(pred, pointclouds_pl, end_points)
#             tf.summary.scalar('loss', loss)

#             # correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
#             # accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
#             # tf.summary.scalar('accuracy', accuracy)

#             # Get training operator
#             learning_rate = get_learning_rate(batch)
#             tf.summary.scalar('learning_rate', learning_rate)
#             if OPTIMIZER == 'momentum':
#                 optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
#             elif OPTIMIZER == 'adam':
#                 optimizer = tf.train.AdamOptimizer(learning_rate)
#             train_op = optimizer.minimize(loss, global_step=batch)
            
#             # Add ops to save and restore all the variables.
#             saver = tf.train.Saver()
            
#         # Create a session
#         config = tf.ConfigProto()
#         config.gpu_options.allow_growth = True
#         config.allow_soft_placement = True
#         config.log_device_placement = False
#         sess = tf.Session(config=config)

#         # Add summary writers
#         #merged = tf.merge_all_summaries()
#         merged = tf.summary.merge_all()
#         train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
#                                   sess.graph)
#         test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

#         # Init variables
#         init = tf.global_variables_initializer()
#         # To fix the bug introduced in TF 0.12.1 as in
#         # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
#         #sess.run(init)
#         sess.run(init, {is_training_pl: True})

#         ops = {'pointclouds_pl': pointclouds_pl,
#                'labels_pl': labels_pl,
#                'is_training_pl': is_training_pl,
#                'pred': pred,
#                'loss': loss,
#                'train_op': train_op,
#                'merged': merged,
#                'step': batch,
#                'pt_2d': end_points['pts_2d']}

#         for epoch in range(MAX_EPOCH):
#             log_string('**** EPOCH %03d ****' % (epoch))
#             sys.stdout.flush()
             
#             train_one_epoch(sess, ops, train_writer)
#             eval_one_epoch(sess, ops, test_writer)
            
#             # Save the variables to disk.
#             if epoch % 10 == 0:
#                 save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
#                 log_string("Model saved in file: %s" % save_path)



# def train_one_epoch(sess, ops, train_writer):
#     """ ops: dict mapping from string to tf ops """
#     is_training = True
    
#     # Shuffle train files
#     train_file_idxs = np.arange(0, len(TRAIN_FILES))
#     np.random.shuffle(train_file_idxs)
    
#     for fn in range(len(TRAIN_FILES)):
#         log_string('----' + str(fn) + '-----')
#         current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
#         current_data = current_data[:,0:NUM_POINT,:]
#         current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
#         current_label = np.squeeze(current_label)
        
#         file_size = current_data.shape[0]
#         num_batches = file_size // BATCH_SIZE
        
#         total_correct = 0
#         total_seen = 0
#         loss_sum = 0
       
#         for batch_idx in range(num_batches):
#             start_idx = batch_idx * BATCH_SIZE
#             end_idx = (batch_idx+1) * BATCH_SIZE
            
#             # Augment batched point clouds by rotation and jittering
#             rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
#             jittered_data = provider.jitter_point_cloud(rotated_data)

#             # # test, kaidong
#             # import pdb
#             # pdb.set_trace()
#             # another_pc_pl, _ = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
#             # dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(ops['pointclouds_pl'], another_pc_pl)
#             # loss = tf.reduce_mean(dists_forward + dists_backward)
#             # loss = loss * 100

#             # jittered_data_cp = copy.deepcopy(jittered_data)

#             # feed_dict = { ops['pointclouds_pl']: jittered_data,
#             #               another_pc_pl: current_data[start_idx:end_idx, :, :], }
#             # loss_val = sess.run([loss], feed_dict=feed_dict)
#             # pdb.set_trace()


#             feed_dict = {ops['pointclouds_pl']: jittered_data,
#                          ops['labels_pl']: current_label[start_idx:end_idx],
#                          ops['is_training_pl']: is_training,}
#             summary, step, _, loss_val, pred_val, pt_2d_val = sess.run([ops['merged'], ops['step'],
#                 ops['train_op'], ops['loss'], ops['pred'], ops['pt_2d']], feed_dict=feed_dict)







#             # # test, kaidong
#             import pdb
#             pdb.set_trace()








#             train_writer.add_summary(summary, step)
#             # pred_val = np.argmax(pred_val, 1)
#             # correct = np.sum(pred_val == current_label[start_idx:end_idx])
#             # total_correct += correct
#             total_seen += BATCH_SIZE
#             loss_sum += loss_val
        
#         log_string('mean loss: %f' % (loss_sum / float(num_batches)))
#         # log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
# def eval_one_epoch(sess, ops, test_writer):
#     """ ops: dict mapping from string to tf ops """
#     is_training = False
#     # total_correct = 0
#     total_seen = 0
#     loss_sum = 0
#     total_seen_class = [0 for _ in range(NUM_CLASSES)]
#     # total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
#     for fn in range(len(TEST_FILES)):
#         log_string('----' + str(fn) + '-----')
#         current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
#         current_data = current_data[:,0:NUM_POINT,:]
#         current_label = np.squeeze(current_label)
        
#         file_size = current_data.shape[0]
#         num_batches = file_size // BATCH_SIZE
        
#         for batch_idx in range(num_batches):
#             start_idx = batch_idx * BATCH_SIZE
#             end_idx = (batch_idx+1) * BATCH_SIZE

#             feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
#                          ops['labels_pl']: current_label[start_idx:end_idx],
#                          ops['is_training_pl']: is_training}
#             summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
#                 ops['loss'], ops['pred']], feed_dict=feed_dict)
#             # pred_val = np.argmax(pred_val, 1)
#             # correct = np.sum(pred_val == current_label[start_idx:end_idx])
#             # total_correct += correct
#             total_seen += BATCH_SIZE
#             loss_sum += (loss_val*BATCH_SIZE)
#             for i in range(start_idx, end_idx):
#                 l = current_label[i]
#                 total_seen_class[l] += 1
#                 # total_correct_class[l] += (pred_val[i-start_idx] == l)
            
#     log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
#     # log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
#     # log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
         


# if __name__ == "__main__":
#     train()
#     LOG_FOUT.close()
