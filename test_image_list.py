# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import os
import logging

from utils.misc_utils import *
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box, draw_demo_img
from utils.eval_utils import *

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
parser.add_argument("--use_gt", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use ground truth to calculate error.")
parser.add_argument("--nV", type=int, default=9,
                    help="Whether to use ground truth to calculate error.")

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
ref_corners = np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)),dtype='float32')
points = np.concatenate((np.array([0.0, 0.0, 0.0, 1.0]).reshape(4, 1), corners3D), axis=1)
diam = calc_pts_diameter(np.array(mesh.vertices))
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

    #Error calculation stats
    eps = 1e-5
    testing_error_trans = 0.0
    testing_error_angle = 0.0
    testing_error_pixel = 0.0
    preds_corners2D = []
    gts_corners2D = []
    errs_corner2D = []
    errs_trans = []
    errs_angle = []
    errs_2d = []
    errs_3d = []
    count = 0

    error_file = open('error.txt', 'w')
    error_file.write("filename translation translation_error \n")
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

        rot, trans, transform = solve_pnp(x_, y_, conf_, ref_corners, selected_, width, height)

        if transform is not None:
            intrinsics = np.array(get_camera_intrinsic(), dtype=np.float32)
            bbox_3d = compute_projection(points, transform, intrinsics)
            corners2D_pr = np.transpose(bbox_3d)

            img_ori = draw_demo_img(img_ori, corners2D_pr, (0, 0, 255))

            if args.use_gt:
                label_file = line.replace('images', 'labels').replace('.png', '.txt').replace(
				'.jpg', '.txt').replace('.jpeg', '.txt').strip()
                if os.path.isfile(label_file):
                    target = open(label_file, 'r').readline().split(' ')
                    target = [float(x) for x in target]
                    box_gt = np.array(target[1:19]).reshape(args.nV, 2)
                    box_gt[:, 0] = box_gt[:, 0] * width
                    box_gt[:, 1] = box_gt[:, 1] * height

                    # Compute corner prediction error
                    corner_norm = np.linalg.norm(box_gt - corners2D_pr, axis=1)
                    corner_dist = np.mean(corner_norm)
                    errs_corner2D.append(corner_dist)

                    # Compute [R|t] by pnp
                    R_gt, t_gt = pnp(
                        np.array(ref_corners, dtype='float32'), box_gt, np.array(intrinsics, dtype='float32'))

                    # Compute translation error
                    trans_dist = np.sqrt(np.sum(np.square(t_gt - trans)))

                    trans_pred = np.sqrt(np.sum(np.square(trans)))
                    if trans_pred > 10:
                        continue

                    errs_trans.append(trans_dist)

                    # Compute angle error
                    angle_dist = calcAngularDistance(R_gt, rot)
                    errs_angle.append(angle_dist)

                    # Compute pixel error
                    Rt_gt = np.concatenate((R_gt, t_gt), axis=1)
                    Rt_pr = np.concatenate((rot, trans), axis=1)
                    proj_2d_gt = compute_projection(vertices, Rt_gt, intrinsics)
                    proj_2d_pred = compute_projection(vertices, Rt_pr, intrinsics)
                    norm = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
                    pixel_dist = np.mean(norm)
                    errs_2d.append(pixel_dist)

                    # print('Pixel Dist: ', pixel_dist)

                    # Compute 3D distances
                    transform_3d_gt = compute_transformation(vertices, Rt_gt)
                    transform_3d_pred = compute_transformation(vertices, Rt_pr)
                    norm3d = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
                    vertex_dist = np.mean(norm3d)
                    errs_3d.append(vertex_dist)

                    # Sum errors
                    error_file.write('%s %f %f %f %f\n' % (line, trans[0], trans[1], trans[2], trans_dist))

                    testing_error_trans += trans_dist
                    testing_error_angle += angle_dist
                    testing_error_pixel += pixel_dist
                    count = count + 1
                else:
                    print(' No label for this image')


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

    px_threshold = 100
    acc = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d) + eps)
    acc5cm5deg = len(np.where((np.array(errs_trans) <= 0.05) & (np.array(errs_angle) <= 5))[0]) * 100. / (
            len(errs_trans) + eps)
    acc3d10 = len(np.where(np.array(errs_3d) <= diam * 0.1)[0]) * 100. / (len(errs_3d) + eps)
    acc5cm5deg = len(np.where((np.array(errs_trans) <= 0.05) & (np.array(errs_angle) <= 5))[0]) * 100. / (
            len(errs_trans) + eps)
    corner_acc = len(np.where(np.array(errs_corner2D) <= px_threshold)[0]) * 100. / (len(errs_corner2D) + eps)
    mean_err_2d = np.mean(errs_2d)
    mean_corner_err_2d = np.mean(errs_corner2D)

    # Print test statistics
    logging.error('Results of {}'.format('Aqua'))
    logging.error('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))
    logging.error('   Acc using 10% threshold - {} vx 3D Transformation = {:.2f}%'.format(diam * 0.1, acc3d10))
    logging.error('   Acc using 5 cm 5 degree metric = {:.2f}%'.format(acc5cm5deg))
    logging.error("   Mean 2D pixel error is %f, Mean vertex error is %f, mean corner error is %f" % (
        mean_err_2d, np.mean(errs_3d), mean_corner_err_2d))
    logging.error('   Translation error: %f m, angle error: %f degree, pixel error: % f pix' % (
        testing_error_trans / count, testing_error_angle / count, testing_error_pixel / count))

    error_file.close()
    if args.save_video:
        videoWriter.release()


