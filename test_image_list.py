# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2

from utils.misc_utils import *
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box, draw_demo_img

from model import yolov3
from tqdm import tqdm
from region_loss import RegionLoss

from utils.meshply import MeshPly

parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("--image_list", type=str,
                    help="The path of the input image.", default='/media/bjoshi/ssd-data/deepcl-data/pool/pool_coco.txt')
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--class_name_path", type=str, default="./data/aqua.names",
                    help="The path of the class names.")
parser.add_argument("--checkpoint_dir", type=str, default="/home/bjoshi/checkpoint",
                    help="The path of the weights to restore.")
parser.add_argument("--save_video", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to save the video detection results.")
parser.add_argument("--mesh_path", type=str, default='/home/bjoshi/singleshotv3-tf/aqua_glass_removed.ply',
                    help="Aqua Mesh Model")

args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

lines = open(args.image_list, 'r').readlines()

height = 600
width = 800

mesh = MeshPly(args.mesh_path)
vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
corners3D = get_3D_corners(vertices)
gt_corners = np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)),dtype='float32')
points = np.concatenate((np.array([0.0, 0.0, 0.0, 1.0]).reshape(4, 1), corners3D), axis=1)

if args.save_video:
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter('video_result_pool6.mp4', fourcc, 20, (width, height))

with tf.Session(config=config) as sess:
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    region_loss = RegionLoss(1, num_classes=1)

    yolo_model = yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    yolo_features = [pred_feature_maps[0], pred_feature_maps[1], pred_feature_maps[2]]
    region_features = [pred_feature_maps[3], pred_feature_maps[4], pred_feature_maps[5]]


    pred_boxes, pred_confs, pred_probs = yolo_model.predict(yolo_features)

    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=1, score_thresh=0.3,
                                    nms_thresh=0.45)

    x, y, conf, selected = region_loss.predict(region_features,  boxes, scores, num_classes=1)

    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
    saver.restore(sess, checkpoint)

    for line in tqdm(lines):
        line = line.strip()
        # print(line)
        img_ori = cv2.imread(line)

        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(args.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.

        boxes_, scores_, labels_, x_, y_, conf_, selected_ = sess.run([boxes, scores, labels, x, y, conf, selected ], feed_dict={input_data: img})

        transform = solve_pnp(x_, y_, conf_, gt_corners, selected_, width, height)

        if transform is not None:
            intrinsics = np.array(get_camera_intrinsic(), dtype=np.float32)
            bbox_3d = compute_projection(points, transform, intrinsics)
            corners2D_pr = np.transpose(bbox_3d)
            # print(transform)
        # corners2D_gt = np.array(np.reshape(box_gt[:18], [9, 2]), dtype='float32')
        # corners2D_pr = np.array(np.reshape(bbox3d[:18], [9, 2]), dtype='float32')
        # corners2D_gt[:, 0] = corners2D_gt[:, 0] * 416
        # corners2D_gt[:, 1] = corners2D_gt[:, 1] * 416
        # corners2D_pr[:, 0] = corners2D_pr[:, 0] * width
        # corners2D_pr[:, 1] = corners2D_pr[:, 1] * height

            img_ori = draw_demo_img(img_ori, corners2D_pr, (0, 0, 255))


        # print('*' * 30)
        # print("scores:")
        # print(scores_)

        boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
        boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))

        # print("Print Boxes", boxes_)
        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            plot_one_box(img_ori, [x0, y0, x1, y1],
                         label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=(0, 255, 0))
        # cv2.imshow('Image', img_ori)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if args.save_video:
            videoWriter.write(img_ori)

    if args.save_video:
        videoWriter.release()


