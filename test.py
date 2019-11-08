# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import logging
from tqdm import trange
import cv2
import args

from utils.data_utils import get_batch_data
from utils.misc_utils import shuffle_and_overwrite, make_summary, config_learning_rate, config_optimizer, AverageMeter
from utils.eval_utils import evaluate_on_cpu, evaluate_on_gpu, get_preds_gpu, voc_eval, parse_gt_rec
from utils.nms_utils import gpu_nms
from region_loss import *

from model import yolov3

def draw_demo_img(img, projectpts, color = (0, 255, 0)):

    vertices = []
    for i in range(9):
        x = projectpts[i][0]
        y = projectpts[i][1]
        coordinates = (int(x),int(y))
        vertices.append(coordinates)
        cv2.circle(img, coordinates, 1, (0, 255, 255), -1)

    # print(vertices)
    cv2.line(img, vertices[1], vertices[2], color, 2)
    cv2.line(img, vertices[1], vertices[3], color, 2)
    cv2.line(img, vertices[1], vertices[5], color, 2)
    cv2.line(img, vertices[2], vertices[6], color, 2)
    cv2.line(img, vertices[2], vertices[4], color, 2)
    cv2.line(img, vertices[3], vertices[4], color, 2)
    cv2.line(img, vertices[3], vertices[7], color, 2)
    cv2.line(img, vertices[4], vertices[8], color, 2)
    cv2.line(img, vertices[5], vertices[6], color, 2)
    cv2.line(img, vertices[5], vertices[7], color, 2)
    cv2.line(img, vertices[6], vertices[8], color, 2)
    cv2.line(img, vertices[7], vertices[8], color, 2)

    return img



# setting loggers

tf.enable_eager_execution()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename=args.progress_log_path, filemode='w')



val_dataset = tf.data.TextLineDataset(args.train_file)
val_dataset = val_dataset.batch(1)
val_dataset = val_dataset.map(
    lambda x: tf.py_func(get_batch_data,
                         inp=[x, args.class_num, args.img_size, args.anchors, 'train', False, False, args.letterbox_resize],
                         Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=args.num_threads
)
val_dataset.prefetch(args.prefetech_buffer)

iterator = tf.data.Iterator.from_structure(val_dataset.output_types, val_dataset.output_shapes)
train_init_op = iterator.make_initializer(val_dataset)
val_init_op = iterator.make_initializer(val_dataset)

# # get an element from the chosen dataset iterator
image_ids, image, y_true_13, y_true_26, y_true_52, slabels = iterator.get_next()
# print(slabels)
# batch = image.shape[0]
# for i in range(batch):
#     slabel = slabels[i][0]
#     predictions = slabel[1:].numpy().reshape(9,2)
#     predictions[:, 0] = predictions[:, 0] * 416
#     predictions[:, 1] = predictions[:, 1] * 416
#
#     # print(predictions.shape)
#     img = image[i].numpy()
#     img = img * 255.
#     img = img.astype(np.uint8)
#     # cv2.imwrite('0.png', img)
#     img = draw_demo_img(img, predictions)
#     # cv2.imwrite('0.png', img)
#     cv2.imshow('Image', img)
#     cv2.waitKey(0)
y_true = [y_true_13, y_true_26, y_true_52]

image_ids.set_shape([None])
image.set_shape([None, None, None, 3])
for y in y_true:
    y.set_shape([None, None, None, None, None])

##################
# Model definition
##################
yolo_model = yolov3(args.class_num, args.anchors, args.use_label_smooth, args.use_focal_loss, args.batch_norm_decay, args.weight_decay, use_static_shape=False)
with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(image, is_training=True)
yolo_preds = [pred_feature_maps[0],pred_feature_maps[1], pred_feature_maps[2]]
region_preds = [pred_feature_maps[3],pred_feature_maps[4], pred_feature_maps[5]]

loss = yolo_model.compute_loss(yolo_preds, y_true)
# print(yolo_preds)

region_loss = RegionLoss(1, num_classes=1)

loss = region_loss.compute_loss(region_preds, slabels)


