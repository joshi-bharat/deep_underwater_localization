import time
from utils import *
import tensorflow as tf
import numpy as np
from utils.misc_utils import get_bbox_mask

class PoseRegressionLoss():
    def __init__(self, batch_size, num_classes=1, nV=9):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.coord_scale = 1
        self.noobject_scale = 0.1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.nV = nV

    def pose_regression_loss(self, output, target, bbox_mask):
        # Parameters
        #Shape of prediction [ b, 13, 13, 20]

        nB = output.shape[0]
        nH = output.shape[1]
        nW = output.shape[2]
        output = tf.transpose(output, [0, 3, 1, 2])

        x = output[:,0:self.nV,...]
        y = output[:,self.nV:2*self.nV,...]
        conf = tf.sigmoid(output[:,2*self.nV:3*self.nV,...])

        grid_x = tf.range(nH, dtype=tf.int32)
        grid_y = tf.range(nW, dtype=tf.int32)


        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)

        grid_x = tf.cast(grid_x, tf.float32)
        grid_y = tf.cast(grid_y, tf.float32)

        # print(grid_x)
        # print(grid_y)
        #Shape of predx [b, 9, h, w]
        predx = (x + grid_x)/tf.cast(nW, tf.float32)
        predy = (y + grid_y)/tf.cast(nH, tf.float32)

        nCorrect, bbox_masks, conf_mask, tconf, targetx, targety  = self.build_targets(predx, predy, target, bbox_mask, grid_x, grid_y)

        nProposals = tf.count_nonzero(conf > 0.5)
        # conf_mask = tf.sqrt(conf_mask)

        coord_mask = tf.transpose(bbox_masks, [0, 3, 1, 2])

        predx = predx * tf.cast(nW, tf.float32) - grid_x
        predy = predy * tf.cast(nH, tf.float32) - grid_y
        loss_x = tf.reduce_sum(tf.abs(predx - targetx) * coord_mask)
        loss_y = tf.reduce_sum(tf.abs(predy - targety) * coord_mask)

        target_conf = tf.transpose(tconf, [0, 3, 1, 2])
        conf_mask = tf.transpose(conf_mask, [0, 3, 1, 2])
        loss_conf = tf.reduce_sum(tf.abs(conf - target_conf) * conf_mask)

        return nCorrect, nProposals, loss_x, loss_y, loss_conf

    def compute_loss(self, region_preds, slabels, bbox_mask):
        nCorrect, nProposals, loss_x, loss_y, loss_conf, loss = 0, 0, 0, 0, 0, 0
        # print(region_preds)
        for i in range(len(region_preds)):  #Change this later
            # print(i)
            pred = tf.reshape(region_preds[i],[self.batch_size, 2**i * 13, 2**i * 13, self.nV*3+1])
            total_loss = self.pose_regression_loss(pred, slabels, bbox_mask[i])
            nCorrect += total_loss[0]
            nProposals += total_loss[1]
            loss_x += total_loss[2]
            loss_y += total_loss[3]
            loss_conf += total_loss[4]

        loss = loss_x + loss_y + loss_conf
        return [loss, loss_x, loss_y, loss_conf, nProposals, nCorrect]


    def predict(self, outputs, bboxes, scores, num_classes=1):

        def reorg(output):
            # Parameters
            batch = output.shape[0]
            h = output.shape[1]
            w = output.shape[2]

            # use some broadcast tricks to get the mesh coordinates
            grid_x = tf.range(h, dtype=tf.int32)
            grid_y = tf.range(w, dtype=tf.int32)

            grid_x, grid_y = tf.meshgrid(grid_x, grid_y)

            grid_x = tf.cast(grid_x, tf.float32)
            grid_y = tf.cast(grid_y, tf.float32)

            conf = output[..., 2*self.nV:3*self.nV]

            output = tf.transpose(output, [0, 3, 1, 2])
            x = output[:, 0:self.nV, ...]
            y = output[:, self.nV:2*self.nV, ...]

            predx = (x + grid_x) / tf.cast(w, tf.float32)
            predy = (y + grid_y) / tf.cast(h, tf.float32)

            predx = tf.transpose(predx, [0, 2, 3, 1])
            predy = tf.transpose(predy, [0, 2, 3, 1])

            #Ignoring batch size and assuming single image
            #Need to fix later

            predx = tf.reshape(predx, [h, w, self.nV])
            predy = tf.reshape(predy, [h, w, self.nV])
            conf = tf.reshape(conf, [h, w, self.nV])

            return predx, predy, conf

        bbox_masks = get_bbox_mask(bboxes)

        for i in range(len(outputs)):
            reorg_results = [reorg(output) for output in outputs]

        x_list, y_list, confs_list = [], [], []

        if bbox_masks is not None:
            for i, result in enumerate(reorg_results):
                x, y, conf = result
                mask = bbox_masks[i]
                # mask = tf.expand_dims(mask, axis=0)
                # mask = tf.tile(mask, [self.batch_size, 1, 1])

                # print(conf.shape)
                conf = tf.sigmoid(conf)
                pred_x = tf.boolean_mask(x, mask)
                pred_y = tf.boolean_mask(y, mask)
                pred_conf = tf.boolean_mask(conf, mask)

                x_list.append(pred_x)
                y_list.append(pred_y)
                confs_list.append(pred_conf)

        else:
            for i, result in enumerate(reorg_results):
                x, y, conf = result
                w = x.shape[0]
                h = x.shape[1]

                x = tf.reshape(x, [h*w, self.nV])
                y = tf.reshape(y, [h*w, self.nV])
                conf = tf.reshape(conf, [h * w, self.nV])
                x_list.append(x)
                y_list.append(y)
                confs_list.append(conf)

        # collect results on three scales
        # take 416*416 input image for example:
        # shape: [inside_masks, self.nV]
        pred_x = tf.concat(x_list, axis=0)
        pred_y = tf.concat(y_list, axis=0)
        pred_conf = tf.concat(confs_list, axis=0)

        total_max_count = pred_x.shape[0]
        mean_x  = tf.reduce_mean(pred_x, axis=1)    #average x position
        mean_y = tf.reduce_mean(pred_y, axis=1)     #average y position
        mean_conf = tf.reduce_mean(pred_conf, axis=1)   #average 2D confs

        max_conf_idx = tf.arg_max(mean_conf, 0)

        center_xy = tf.transpose(tf.stack([mean_x, mean_y]), [1, 0])
        ref_xy = tf.tile(tf.reshape(center_xy[max_conf_idx], [1, -1]), [total_max_count, 1])
        selected = tf.linalg.norm(center_xy - ref_xy, axis=1) < 0.3

        return pred_x, pred_y, pred_conf, selected

    def build_targets(self, pred_x, pred_y, target, bbox_mask, grid_x, grid_y):

        nB = pred_x.shape[0]
        nH = pred_x.shape[2]
        nW = pred_x.shape[3]

        nAnchors = nH * nW

        conf_mask = tf.ones([nB, nH, nW, self.nV])

        targets = tf.reshape(target, [-1, 2*self.nV + 1])
        targets = targets[:,1:2*self.nV + 1]
        # print(targets)
        target_x = targets[:, ::2]
        target_y = targets[:, 1::2]

        target_x = tf.expand_dims(target_x, axis=2)
        target_y = tf.expand_dims(target_y, axis=2)

        pred_x = tf.reshape(pred_x, [-1, self.nV, nAnchors])
        pred_y = tf.reshape(pred_y, [-1, self.nV, nAnchors])

        target_x = tf.tile(target_x, [1,1,nAnchors])
        target_y = tf.tile(target_y, [1,1,nAnchors])

        cur_confs = self.corner_confidences9(pred_x, target_x, pred_y, target_y)

        cur_confs = tf.reshape(cur_confs, [nB, nH, nW, self.nV])

        bbox_masks = tf.expand_dims(bbox_mask, axis=3)
        bbox_masks = tf.tile(bbox_masks, [1,1,1, self.nV])

        conf_noobj_mask = conf_mask * self.noobject_scale * tf.cast(cur_confs <= self.thresh, tf.float32) *\
                          tf.cast(tf.logical_not(tf.cast(bbox_masks, tf.bool)), tf.float32)

        #removing the noobj mask, not sure if it will improve the results
        conf_mask = conf_mask * bbox_masks * self.object_scale + conf_noobj_mask
        cur_confs = cur_confs * bbox_masks

        target_x = tf.reshape(target_x, [nB, self.nV, nW, nH])
        target_y = tf.reshape(target_y, [nB, self.nV, nW, nH])

        targetx = target_x * tf.cast(nW, tf.float32) - grid_x
        targety = target_y * tf.cast(nH, tf.float32) - grid_y

        nCorrect = tf.count_nonzero(cur_confs > 0.5)

        return nCorrect, bbox_masks, conf_mask, cur_confs, targetx, targety

    def corner_confidences9(self, pred_x, target_x, pred_y, target_y, sharpness=6):


        distx = pred_x - target_x
        disty = pred_y - target_y

        # Convert to [b, 169, 9]
        distx = tf.transpose(distx , [0, 2, 1])
        disty = tf.transpose(disty, [0, 2, 1])

        distx = tf.square(distx)
        disty = tf.square(disty)


        dist = distx + disty

        # print(dist)
        conf = tf.exp(sharpness * -1.0 * dist)
        # print(conf)

        return conf
