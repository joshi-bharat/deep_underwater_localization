# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2

from utils.misc_utils import *
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box, draw_demo_img_corners
from utils.data_aug import letterbox_resize

from model import yolov3
from tqdm import tqdm
from pose_loss import PoseRegressionLoss

from utils.meshply import MeshPly

parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("--input_video", type=str,
                    help="The path of the input image.", default='/media/bjoshi/data1/Gopro-hero7/GH010136.MP4')
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--class_name_path", type=str, default="./data/aqua.names",
                    help="The path of the class names.")
parser.add_argument("--checkpoint_dir", type=str, default="/home/bjoshi/deep_localization/checkpoint",
                    help="The path of the weights to restore.")
parser.add_argument("--save_video", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to save the video detection results.")
parser.add_argument("--mesh_path", type=str, default='/home/bjoshi/singleshotv3-tf/aqua_glass_removed.ply',
                    help="Aqua Mesh Model")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--nV", type=int, default=8,
                    help="Number of corner points used for PnP.")
parser.add_argument("--rectify", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Rectify images")

args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

vid = cv2.VideoCapture(args.input_video)
video_frame_cnt = int(vid.get(7))
video_width = int(vid.get(3))
video_height = int(vid.get(4))
video_fps = int(vid.get(5))


mesh = MeshPly(args.mesh_path)
vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
corners3D = get_3D_corners(vertices)
gt_corners = np.array(np.transpose(corners3D[:3, :]),dtype='float32')
points = np.concatenate(( corners3D, np.array([0.0, 0.0, 0.0, 1.0]).reshape(4, 1)), axis=1)

if args.save_video:
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter('result_gopro_10136.mp4', fourcc, 30, (video_width, video_height))

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

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=1, score_thresh=0.3,
                                    nms_thresh=0.4)

    x, y, conf, selected = pose_loss.predict(pose_features,  boxes, scores, num_classes=1)

    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
    saver.restore(sess, checkpoint)

    error_count = 0
    intrinsics = get_gopro_instrinsic()
    dist = get_gopro_distortion()

    for j in tqdm(range(video_frame_cnt)):
        ret, img_ori = vid.read()
        if img_ori is None:
            continue


        height_ori, width_ori = img_ori.shape[:2]

        if args.rectify:
            map1, map2 = cv2.initUndistortRectifyMap(intrinsics, dist, None, None, (width_ori, height_ori),
                                                     cv2.CV_32FC1)
            img_ori = cv2.remap(img_ori, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # cv2.imshow("Image", img_ori)
        # cv2.waitKey(0)

        if args.letterbox_resize:
            img_resize, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
        else:
            height_ori, width_ori = img_ori.shape[:2]
            img_resize = cv2.resize(img_ori, tuple(args.new_size))

        # cv2.imshow("Image", img_resize)
        # cv2.waitKey(0)

        img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.

        boxes_, scores_, labels_, x_, y_, conf_, selected_ = sess.run([boxes, scores, labels, x, y, conf, selected ], feed_dict={input_data: img})

        if len(boxes_) == 0:
            error_count += 1
            continue

        if args.letterbox_resize:
            x_ = (x_ * args.new_size[0] - dw ) / resize_ratio
            y_ = (y_ * args.new_size[1] - dh ) / resize_ratio
        else:
            x_ = x_ * args.new_size[0]
            y_ = y_ * args.new_size[1]

        rot, trans, transform = solve_pnp(x_, y_, conf_, gt_corners, selected_, intrinsics, bestCnt=12, nV=8)

        if transform is not None:
    #         print(trans)
            bbox_3d = compute_projection(points, transform, intrinsics)
            corners2D_pr = np.transpose(bbox_3d)

    #         # print(corners2D_pr)
            try:
                img_ori = draw_demo_img_corners(img_ori, corners2D_pr, (0, 0, 255), nV=8, thickness=16)
            except:
                print("Something Went Wrong")

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
                         label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=(0, 255, 0), line_thickness=16)


        if args.save_video:
            videoWriter.write(img_ori)

    vid.release()
    if args.save_video:
        videoWriter.release()
