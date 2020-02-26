# coding: utf-8

import numpy as np
import tensorflow as tf
import random

from tensorflow.core.framework import summary_pb2
import cv2

def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.average = self.sum / float(self.count)


def parse_anchors(anchor_path):
    '''
    parse anchors.
    returned data: shape [N, 2], dtype float32
    '''
    anchors = np.reshape(np.asarray(open(anchor_path, 'r').read().split(','), np.float32), [-1, 2])
    return anchors


def read_class_names(class_name_path):
    names = {}
    with open(class_name_path, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def shuffle_and_overwrite(file_name):
    content = open(file_name, 'r').readlines()
    random.shuffle(content)
    with open(file_name, 'w') as f:
        for line in content:
            f.write(line)


def update_dict(ori_dict, new_dict):
    if not ori_dict:
        return new_dict
    for key in ori_dict:
        ori_dict[key] += new_dict[key]
    return ori_dict


def list_add(ori_list, new_list):
    for i in range(len(ori_list)):
        ori_list[i] += new_list[i]
    return ori_list


def load_weights(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    param:
        var_list: list of network variables.
        weights_file: name of the binary file.
    """
    with open(weights_file, "rb") as fp:
        np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        if 'Conv' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'BatchNorm' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'Conv' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr +
                                           bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(
                tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops


def config_learning_rate(args, global_step):
    if args.lr_type == 'exponential':
        lr_tmp = tf.train.exponential_decay(args.learning_rate_init, global_step, args.lr_decay_freq,
                                            args.lr_decay_factor, staircase=True, name='exponential_learning_rate')
        return tf.maximum(lr_tmp, args.lr_lower_bound)
    elif args.lr_type == 'cosine_decay':
        train_steps = (args.total_epoches - float(args.use_warm_up) * args.warm_up_epoch) * args.train_batch_num
        return args.lr_lower_bound + 0.5 * (args.learning_rate_init - args.lr_lower_bound) * \
               (1 + tf.cos(global_step / train_steps * np.pi))
    elif args.lr_type == 'cosine_decay_restart':
        return tf.train.cosine_decay_restarts(args.learning_rate_init, global_step,
                                              args.lr_decay_freq, t_mul=2.0, m_mul=1.0,
                                              name='cosine_decay_learning_rate_restart')
    elif args.lr_type == 'fixed':
        return tf.convert_to_tensor(args.learning_rate_init, name='fixed_learning_rate')
    elif args.lr_type == 'piecewise':
        return tf.train.piecewise_constant(global_step, boundaries=args.pw_boundaries, values=args.pw_values,
                                           name='piecewise_learning_rate')
    else:
        raise ValueError('Unsupported learning rate type!')


def config_optimizer(optimizer_name, learning_rate, decay=0.9, momentum=0.9):
    if optimizer_name == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=momentum)
    elif optimizer_name == 'adam':
        return tf.train.AdamOptimizer(learning_rate)
    elif optimizer_name == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Unsupported optimizer type!')


def get_bbox_mask(bbox, img_size=(416, 416)):

    if bbox is not None:
        return None

    y_true_13 = np.zeros((img_size[1] // 32, img_size[0] // 32), np.float32)
    y_true_26 = np.zeros((img_size[1] // 16, img_size[0] // 16), np.float32)
    y_true_52 = np.zeros((img_size[1] // 8, img_size[0] // 8), np.float32)

    # y_true_13_final = np.zeros((img_size[1] // 32, img_size[0] // 32), np.float32)
    # y_true_26_final = np.zeros((img_size[1] // 16, img_size[0] // 16), np.float32)
    # y_true_52_final = np.zeros((img_size[1] // 8, img_size[0] // 8), np.float32)

    for box in bbox:
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        scale_13_x1 = int(x1 // 32)
        scale_13_y1 = int(y1 // 32)
        scale_13_x2 = int(x2 // 32)
        scale_13_y2 = int(y2 // 32)
        y_true_13[int(scale_13_y1):int(scale_13_y2) + 1, int(scale_13_x1):int(scale_13_x2) + 1] = 1

        scale_26_x1 = int(x1 // 16)
        scale_26_y1 = int(y1 // 16)
        scale_26_x2 = int(x2 // 16)
        scale_26_y2 = int(y2 // 16)
        y_true_26[int(scale_26_y1):int(scale_26_y2) + 1, int(scale_26_x1):int(scale_26_x2) + 1] = 1

        scale_52_x1 = int(x1 // 8)
        scale_52_y1 = int(y1 // 8)
        scale_52_x2 = int(x2 // 8)
        scale_52_y2 = int(y2 // 8)
        y_true_52[int(scale_52_y1):int(scale_52_y2) + 1, int(scale_52_x1):int(scale_52_x2) + 1] = 1

        # y_true_13_final = y_true_13 + y_true_13_final
        # y_true_26_final = y_true_26 + y_true_26_final
        # y_true_52_final = y_true_52 + y_true_52_final

    y_true_13 = tf.convert_to_tensor(y_true_13)
    y_true_26 = tf.convert_to_tensor(y_true_26)
    y_true_52 = tf.convert_to_tensor(y_true_52)

    return y_true_13, y_true_26, y_true_52

def get_3D_corners(vertices):
    min_x = np.min(vertices[0, :])
    max_x = np.max(vertices[0, :])
    min_y = np.min(vertices[1, :])
    max_y = np.max(vertices[1, :])
    min_z = np.min(vertices[2, :])
    max_z = np.max(vertices[2, :])

    corners = np.array([[min_x, min_y, min_z],
                        [min_x, min_y, max_z],
                        [min_x, max_y, min_z],
                        [min_x, max_y, max_z],
                        [max_x, min_y, min_z],
                        [max_x, min_y, max_z],
                        [max_x, max_y, min_z],
                        [max_x, max_y, max_z]])

    corners = np.concatenate((np.transpose(corners), np.ones((1, 8))), axis=0)
    return corners

def get_camera_intrinsic():
	K = np.array([[569.31671203, 0.0, 360.09063137],
	              [0.0, 569.387306625, 301.45327471],
	              [0.0, 0.0, 1.0]])
	return K

def get_old_pool_intrinsics():
	K = np.array([[569.416384877, 0.0, 354.086468692],
	              [0.0, 569.797349037, 308.564486913],
	              [0.0, 0.0, 1.0]])
	return K

def get_gopro_instrinsic():
    K = np.array([2.5768e+03, 0.0, 1.8541e+03,
                  0.0, 2.5976e+03, 1.0693e+03
                 , 0.0, 0.0, 1.0]).astype(np.float32).reshape(3, 3)
    return K

def get_gopro_distortion():
    dist = np.array([-0.1181, 0.1363, 5.4112e-04, -0.0047, 0.0]).astype(np.float32)
    return dist

def compute_projection(points_3D, transformation, internal_calibration):
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :] / camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :] / camera_projection[2, :]
    return projections_2d

def solve_pnp(x, y, conf, gt_corners, selected, intrinsics, bestCnt=12, nV=9):
    xsi = x[selected]
    ysi = y[selected]
    dsi = conf[selected]

    gridCnt = len(xsi)
    assert (gridCnt > 0)
    # choose best N count
    p2d = None
    p3d = None
    candiBestCnt = min(gridCnt, bestCnt)

    for i in range(candiBestCnt):
        bestGrids = dsi.argmax(axis=0)
        validmask = dsi[bestGrids, list(range(nV))] > 0.5
        xsb = xsi[bestGrids, list(range(nV))][validmask]
        ysb = ysi[bestGrids, list(range(nV))][validmask]
        t2d = np.concatenate((xsb.reshape(-1, 1), ysb.reshape(-1, 1)), 1)
        t3d = gt_corners[validmask]
        if p2d is None:
            p2d = t2d
            p3d = t3d
        else:
            p2d = np.concatenate((p2d, t2d), 0)
            p3d = np.concatenate((p3d, t3d), 0)
        dsi[bestGrids, list(range(nV))] = 0

    if(len(p3d)) < 6:
        #will need to select the best one may be but not sure
        print("Not enough points for Ransac")
        return None, None, None
    retval, rot, trans, inliers = cv2.solvePnPRansac(p3d, p2d, intrinsics, None, flags=cv2.SOLVEPNP_EPNP)

    if not retval:
        print("Ransac did not converge")
        return None, None, None

    R = cv2.Rodrigues(rot)[0]  # convert to rotation matrix
    T = trans.reshape(-1, 1)
    rt = np.concatenate((R, T), 1)

    return R, T, rt
