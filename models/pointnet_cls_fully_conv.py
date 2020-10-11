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

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    


    net = tf.reshape(point_cloud, [batch_size, -1])
    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, num_point*2, activation_fn=None, scope='fc3')
    pts_2d = tf.reshape(net, [batch_size, num_point, 1, 2])


    end_points['pts_2d'] = pts_2d



    # decoder
    net = tf_util.conv2d(pts_2d, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv8', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv9', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv10', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 3, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv11', bn_decay=bn_decay)
    net = tf.squeeze(net, axis=[2])

    return net, end_points


def get_loss(pred, label, end_points, reg_weight=0.001):









    # # test, kaidong
    # import pdb
    # pdb.set_trace()











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
        loss = get_loss(outputs[0], tf.zeros((32,1024,3)), outputs[1])