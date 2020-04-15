# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import os
import logging
import time

from utils.misc_utils import *
from utils.plot_utils import get_color_table, plot_one_box, draw_demo_img_corners
from utils.eval_utils import *
from utils.data_utils import letterbox_resize

from model import yolov3
from tqdm import tqdm
from pose_loss import PoseRegressionLoss

from utils.meshply import MeshPly

parser = argparse.ArgumentParser(description="DeepURL: test single image.")
parser.add_argument("--input_image", type=str,
                    help="The path of the input image.", default='data/demo_data/1569602987295597399.jpg')
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--class_name_path", type=str, default="./data/aqua.names",
                    help="The path of the class names.")
parser.add_argument("--checkpoint_dir", type=str, default="/home/bjoshi/deep_localization/checkpoint",
                    help="The path of the weights to restore.")
parser.add_argument("--mesh_path", type=str, default='aqua_glass_removed.ply',
                    help="Aqua Mesh Model")
parser.add_argument("--nV", type=int, default=8,
                    help="Number of corner points used for PnP.")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--save_result", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to save the image detection results.")

args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

height = 600
width = 800

mesh = MeshPly(args.mesh_path)
vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
corners3D = get_3D_corners(vertices)
#for 9 points
#gt_corners = np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)),dtype='float32')

#for 8 points
ref_corners = np.array(np.transpose(corners3D[:3, :]),dtype='float32')
points = np.concatenate(( corners3D, np.array([0.0, 0.0, 0.0, 1.0]).reshape(4, 1)), axis=1)
diam = calc_pts_diameter(np.array(mesh.vertices))

intrinsics = get_camera_intrinsic()
# intrinsics = get_old_pool_intrinsics()
with tf.Session(config=config) as sess:
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    pose_loss = PoseRegressionLoss(1, num_classes=1, nV=args.nV)

    yolo_model = yolov3(args.num_class, args.anchors, nV=args.nV)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    yolo_features = [pred_feature_maps[0], pred_feature_maps[1], pred_feature_maps[2]]
    pose_features = [pred_feature_maps[3], pred_feature_maps[4], pred_feature_maps[5]]


    pred_boxes, pred_confs, pred_probs = yolo_model.predict(yolo_features)

    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=1, score_thresh=0.25,
                                    nms_thresh=0.35)

    x, y, conf, selected = pose_loss.predict(pose_features,  boxes, scores, num_classes=1)

    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
    saver.restore(sess, checkpoint)


    img_ori = cv2.imread(args.input_image)
    # print(filename)
    img_ori = cv2.resize(img_ori, (width, height))

    # cv2.imshow('Image', img_ori)
    # cv2.waitKey(0)
    if args.letterbox_resize:
        img_resize, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
    else:
        height_ori, width_ori = img_ori.shape[:2]
        img_resize = cv2.resize(img_ori, tuple(args.new_size))


    img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.

    boxes_, scores_, labels_, x_, y_, conf_, selected_ = sess.run([boxes, scores, labels, x, y, conf, selected ], feed_dict={input_data: img})

    if args.letterbox_resize:
        x_ = (x_ * args.new_size[0] - dw ) / resize_ratio
        y_ = (y_ * args.new_size[1] - dh ) / resize_ratio
    else:
        x_ = x_ * args.new_size[0]
        y_ = y_ * args.new_size[1]

    if len(boxes_) == 0:
        print('No bounding box detected')

    rot, trans, transform = solve_pnp(x_, y_, conf_, ref_corners, selected_, intrinsics, nV=args.nV)
    if transform is not None:
        intrinsics = np.array(get_camera_intrinsic(), dtype=np.float32)
        bbox_3d = compute_projection(corners3D, transform, intrinsics)
        corners2D_pr = np.transpose(bbox_3d)
        # print(corners2D_pr)

        try:
            # img_resize = draw_demo_img_corners(img_resize, corners2D_pr, (0, 0, 255), nV=8)
            # cv2.imshow("Image", img_resize)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            img_ori = draw_demo_img_corners(img_ori, corners2D_pr, (0, 0, 255), nV=8)
            # cv2.imshow("Image", img_ori)
            # cv2.waitKey(0)
        except:
            print("Something went wrong")

    # rescale the coordinates to the original image
    if args.letterbox_resize:
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
    else:
        boxes_[:, [0, 2]] *= (width_ori / float(args.new_size[0]))
        boxes_[:, [1, 3]] *= (height_ori / float(args.new_size[1]))

    # print("Print Boxes", boxes_)
    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        plot_one_box(img_ori, [x0, y0, x1, y1],
                     label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=(0, 255, 0))

    if args.save_result:
        img_name = args.input_image.split('.')[0]
        save_img = img_name + '_deepurl_result.png'
        cv2.imwrite(save_img, img_ori)
    else:
        cv2.imshow('Image', img_ori)
        k = cv2.waitKey(0) & 0XFF
        cv2.destroyAllWindows()
