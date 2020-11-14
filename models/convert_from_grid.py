import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import input_transform_net, feature_transform_net

from loss import tf_nndistance

def placeholder_inputs(batch_size, size_2dgrid, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    2d_grid_pl = tf.placeholder(tf.int32, shape=(batch_size, size_2dgrid^2, 2))
    return 2d_grid_pl, pointclouds_pl


def get_model(num_mlps, start_end_index, grid_points, point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    mlp_sc = 'mlp_num_{}'


    # create list of tensors
    pts_recon = []
    input_image = tf.expand_dims(point_cloud, -1)


    for n in range(num_mlps):
        with tf.variable_scope( mlp_sc.format(n) ) as sc:
            net = tf_util.conv2d(input_image, 64, [1,2],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='conv1', bn_decay=bn_decay)
            net = tf_util.conv2d(net, 64, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='conv2', bn_decay=bn_decay)
            net = tf_util.conv2d(net, 64, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='conv3', bn_decay=bn_decay)
            net = tf_util.conv2d(net, 128, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='conv4', bn_decay=bn_decay)
            net = tf_util.conv2d(net, 1024, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='conv5', bn_decay=bn_decay)
            net = tf_util.conv2d(net, 3, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='conv6', bn_decay=bn_decay)
            pts_recon += net


    with tf.variable_scope('transform_matrix') as sc:
        transform = points_transform_matrix(net, is_training, start_end_index, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    return net, end_points


def get_loss(pred, label, end_points, reg_weight=0.001):









    # test, kaidong
    import pdb
    pdb.set_trace()











    dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(pred, label)
    loss = tf.reduce_mean(dists_forward+dists_backward)
    end_points['pcloss'] = loss
    return loss*100, end_points



    # """ pred: B*NUM_CLASSES,
    #     label: B, """
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    # classify_loss = tf.reduce_mean(loss)
    # tf.summary.scalar('classify loss', classify_loss)

    # # Enforce the transformation as orthogonal matrix
    # transform = end_points['transform'] # BxKxK
    # K = transform.get_shape()[1].value
    # mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    # mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    # mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    # tf.summary.scalar('mat loss', mat_diff_loss)

    # return classify_loss + mat_diff_loss * reg_weight


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
