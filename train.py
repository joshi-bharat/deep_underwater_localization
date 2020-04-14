

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import logging
from tqdm import trange

import args
from pose_loss import  PoseRegressionLoss
from utils.data_utils import get_batch_data
from utils.misc_utils import shuffle_and_overwrite, make_summary, config_learning_rate, config_optimizer, AverageMeter
from utils.eval_utils import evaluate_on_cpu, evaluate_on_gpu, get_preds_gpu, voc_eval, parse_gt_rec
from utils.nms_utils import gpu_nms

from model import yolov3

# setting loggers
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename=args.progress_log_path, filemode='w')

# setting placeholders
is_training = tf.placeholder(tf.bool, name="phase_train")
handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')
# register the gpu nms operation here for the following evaluation scheme
pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
gpu_nms_op = gpu_nms(pred_boxes_flag, pred_scores_flag, args.class_num, args.nms_topk, args.score_threshold, args.nms_threshold)

##################
# tf.data pipeline
##################
train_dataset = tf.data.TextLineDataset(args.train_file)
train_dataset = train_dataset.shuffle(args.train_img_cnt)
train_dataset = train_dataset.batch(args.batch_size)
train_dataset = train_dataset.map(
    lambda x: tf.py_func(get_batch_data,
                         inp=[x, args.class_num, args.img_size, args.anchors, 'train', args.multi_scale_train, args.use_mix_up, args.letterbox_resize, 10, args.nV],
                         Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=args.num_threads
)
train_dataset = train_dataset.prefetch(args.prefetech_buffer)

iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
train_init_op = iterator.make_initializer(train_dataset)

# get an element from the chosen dataset iterator
image_ids, image, y_true_13, y_true_26, y_true_52, slabels, y_true_13_mask, y_true_26_mask, y_true_52_mask = iterator.get_next()
y_true_mask = [y_true_13_mask, y_true_26_mask, y_true_52_mask]
y_true = [y_true_13, y_true_26, y_true_52]

# tf.data pipeline will lose the data `static` shape, so we need to set it manually
image_ids.set_shape([None])
image.set_shape([None, None, None, 3])
for y in y_true:
    y.set_shape([None, None, None, None, None])

##################
# Model definition
##################
poseregression_loss = PoseRegressionLoss(args.batch_size, num_classes=1, nV=args.nV)
yolo_model = yolov3(args.class_num, args.anchors, args.use_label_smooth, args.use_focal_loss, args.batch_norm_decay, args.weight_decay, use_static_shape=False, nV=args.nV)
with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(image, is_training=is_training)
yolo_features = [pred_feature_maps[0], pred_feature_maps[1], pred_feature_maps[2]]
region_features = [pred_feature_maps[3], pred_feature_maps[4], pred_feature_maps[5]]
# single_shot_features =
loss = yolo_model.compute_loss(yolo_features, y_true)
poseloss = poseregression_loss.compute_loss(region_features, slabels, y_true_mask)
y_pred = yolo_model.predict(yolo_features)

l2_loss = tf.losses.get_regularization_loss()

# setting restore parts and vars to update
saver_to_restore = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=args.restore_include, exclude=args.restore_exclude))
update_vars = tf.contrib.framework.get_variables_to_restore(include=args.update_part)

tf.summary.scalar('yolo_loss/total_loss', loss[0])
tf.summary.scalar('yolo_loss/loss_xy', loss[1])
tf.summary.scalar('yolo_loss/loss_wh', loss[2])
tf.summary.scalar('yolo_loss/loss_conf', loss[3])
tf.summary.scalar('yolo_loss/loss_class', loss[4])
tf.summary.scalar('loss_l2', l2_loss)
tf.summary.scalar('loss_ratio', l2_loss / (loss[0] + poseloss[0]))

tf.summary.scalar('region_loss/total_loss', poseloss[0])
tf.summary.scalar('region_loss/loss_x', poseloss[1])
tf.summary.scalar('region_loss/loss_y', poseloss[2])
tf.summary.scalar('region_loss/loss_conf', poseloss[3])

global_step = tf.Variable(float(args.global_step), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
if args.use_warm_up:
    learning_rate = tf.cond(tf.less(global_step, args.train_batch_num * args.warm_up_epoch), 
                            lambda: args.learning_rate_init * global_step / (args.train_batch_num * args.warm_up_epoch),
                            lambda: config_learning_rate(args, global_step - args.train_batch_num * args.warm_up_epoch))
else:
    learning_rate = config_learning_rate(args, global_step)
tf.summary.scalar('learning_rate', learning_rate)

if not args.save_optimizer:
    saver_to_save = tf.train.Saver()
    saver_best = tf.train.Saver()

optimizer = config_optimizer(args.optimizer_name, learning_rate)

# set dependencies for BN ops
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # apply gradient clip to avoid gradient exploding
    gvs = optimizer.compute_gradients(loss[0] + poseloss[0] + l2_loss, var_list=update_vars)
    clip_grad_var = [gv if gv[0] is None else [
          tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
    train_op = optimizer.apply_gradients(clip_grad_var, global_step=global_step)

if args.save_optimizer:
    print('Saving optimizer parameters to checkpoint! Remember to restore the global_step in the fine-tuning afterwards.')
    saver_to_save = tf.train.Saver(max_to_keep=20)
    saver_best = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    saver_to_restore.restore(sess, args.restore_path)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(args.log_dir, sess.graph)

    print('\n----------- start to train -----------\n')

    best_mAP = -np.Inf

    for epoch in range(args.total_epoches):

        sess.run(train_init_op)
        loss_total, loss_xy, loss_wh, loss_conf, loss_class = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        rloss_total, rloss_x, rloss_y, rloss_conf = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        # print(rloss)
        for i in trange(args.train_batch_num):
            _, summary, __y_pred, __y_true, __loss,  __region_loss, __labels, __global_step, __lr = sess.run(
                [train_op, merged, y_pred, y_true,  loss, poseloss, slabels, global_step, learning_rate],
                feed_dict={is_training: True})

            writer.add_summary(summary, global_step=__global_step)

            rloss_total.update(__region_loss[0])
            rloss_x.update(__region_loss[1])
            rloss_y.update(__region_loss[2])
            rloss_conf.update(__region_loss[3])

            if __global_step % args.print_step == 0 and __global_step > 0:

                info = "Epoch: {}, global_step: {} | loss: total: {:.2f}, x: {:.2f}, y: {:.2f}, conf: {:.2f} | ".format(
                        epoch, int(__global_step), rloss_total.average, rloss_x.average, rloss_y.average, rloss_conf.average,)
                print(info)
                logging.info(info)

        # NOTE: this is just demo. You can set the conditions when to save the weights.
        if epoch % args.save_epoch == 0 and epoch > 0:
            # if loss_total.average <= 2.:
            saver_to_save.save(sess, args.save_dir + 'model-epoch_{}'.format(epoch))
