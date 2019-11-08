import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *
import tensorflow as tf
import numpy as np

def corner_confidences9(gt_corners, pr_corners, th=80, sharpness=2, im_width=416, im_height=416):
    ''' gt_corners: Ground-truth 2D projections of the 3D bounding box corners, shape: (16 x nA), type: torch.FloatTensor
        pr_corners: Prediction for the 2D projections of the 3D bounding box corners, shape: (16 x nA), type: torch.FloatTensor
        th        : distance threshold, type: int
        sharpness : sharpness of the exponential that assigns a confidence value to the distance
        -----------
        return    : a torch.FloatTensor of shape (nA,) with 9 confidence values
    '''
    # shape = gt_corners.size()
    nA = gt_corners.shape[2]
    dist = tf.cast(gt_corners, tf.float32) - pr_corners

    dist = tf.transpose(dist, [0, 2, 1])
    # Convert to [b, 169, 18]
    dist = tf.reshape(dist, [-1, nA, 9 , 2])
    #Dims - [b ,169, 9, 2]

    distx = dist[...,0] * im_width
    disty = dist[...,1] * im_height

    distx =  tf.expand_dims(distx, 3)
    disty = tf.expand_dims(disty, 3)

    dist = tf.concat([distx, disty], axis=-1)
    thresh = tf.constant([th], dtype=tf.float32)
    distthresh = tf.expand_dims(thresh, 1)
    distthresh = tf.tile(distthresh, [nA, 9])
    #Changing dimes - [b, nA, 9]
    dist = tf.sqrt(tf.reduce_sum(tf.square(dist), axis=3))
    eps = 1e-5
    mask = (dist < distthresh)
    conf = tf.exp(sharpness*(1 - dist/distthresh))-1  # mask * (torch.exp(math.log(2) * (1.0 - dist/rrt)) - 1)
    conf0 = tf.exp(sharpness*(1 - tf.zeros_like(conf))) - 1 + eps
    conf = conf / conf0
    conf = conf * tf.cast(mask, dtype=tf.float32)  # [b, nA , 9]
    mean_conf = tf.reduce_mean(conf, axis=-1) # [b, nA]
    return mean_conf

def corner_confidence9(gt_corners, pr_corners, th=80, sharpness=2, im_width=416, im_height=416):
    ''' gt_corners: Ground-truth 2D projections of the 3D bounding box corners, shape: (18,) type: list
        pr_corners: Prediction for the 2D projections of the 3D bounding box corners, shape: (18,), type: list
        th        : distance threshold, type: int
        sharpness : sharpness of the exponential that assigns a confidence value to the distance
        -----------
        return    : a list of shape (9,) with 9 confidence values
    '''
    dist = tf.cast(gt_corners, tf.float32) - pr_corners
    dist = tf.reshape(dist, [9 , 2])

    distx = dist[...,0] * im_width
    disty = dist[...,1] * im_height

    distx =  tf.expand_dims(distx, 1)
    disty = tf.expand_dims(disty, 1)

    dist = tf.concat([distx, disty], axis=-1)

    eps = 1e-5
    dist = tf.sqrt(tf.reduce_sum(tf.square(dist), axis=1))
    mask  = dist < th
    conf  = tf.exp(sharpness * (1.0 - dist/th)) - 1
    conf0 = tf.exp(tf.cast(sharpness, tf.float32)) - 1 + eps
    conf  = conf / conf0

    conf  = tf.cast(mask, tf.float32) * conf
    mean_conf = tf.reduce_mean(conf)
    return mean_conf

def build_targets(pred_corners, target, nB, nH, nW, noobject_scale, object_scale, sil_thresh):
    # nB = target.shape[0]

    conf_mask = np.ones([nB, nH, nW]) * noobject_scale
    coord_mask = tf.zeros([nB * nH * nW])
    cls_mask = tf.zeros([nB * nH * nW])
    tx0 = tf.zeros([nB * nH * nW])
    ty0 = tf.zeros([nB * nH * nW])
    tx1 = tf.zeros([nB * nH * nW])
    ty1 = tf.zeros([nB * nH * nW])
    tx2 = tf.zeros([nB * nH * nW])
    ty2 = tf.zeros([nB * nH * nW])
    tx3 = tf.zeros([nB * nH * nW])
    ty3 = tf.zeros([nB * nH * nW])
    tx4 = tf.zeros([nB * nH * nW])
    ty4 = tf.zeros([nB * nH * nW])
    tx5 = tf.zeros([nB * nH * nW])
    ty5 = tf.zeros([nB * nH * nW])
    tx6 = tf.zeros([nB * nH * nW])
    ty6 = tf.zeros([nB * nH * nW])
    tx7 = tf.zeros([nB * nH * nW])
    ty7 = tf.zeros([nB * nH * nW])
    tx8 = tf.zeros([nB * nH * nW])
    ty8 = tf.zeros([nB * nH * nW])
    tconf = tf.zeros([nB * nH * nW])
    tcls = tf.zeros([nB * nH * nW])

    nAnchors = nH * nW

    cur_confs = tf.zeros([nB, nAnchors])

    targets = tf.reshape(target, [-1,19])
    targets = tf.reshape(targets[:, 1:19], [-1, 18, 1])
    cur_gt_corners = tf.tile(targets, [1,1, nAnchors])
    cur_confs = tf.maximum(cur_confs, corner_confidences9(cur_gt_corners, pred_corners))
    cur_confs = tf.reshape(cur_confs, [nB, nH, nW])
    conf_mask = conf_mask * tf.cast( cur_confs <= sil_thresh, tf.float32)

    conf_mask = tf.reshape(conf_mask, [nB* nH* nW])

    # print(conf_mask)
    nCorrect = 0

    gt_box = tf.reshape(target, [nB, 19])
    gi = tf.cast(gt_box[:, 1] * tf.cast(nW, tf.float32), tf.int32)
    gj = tf.cast(gt_box[:, 2] *  tf.cast(nH, tf.float32), tf.int32)

    pred_corners = tf.transpose(pred_corners, [0, 2, 1]) # make [b,169, 18]

    nw = tf.cast(nW, tf.float32)
    nh = tf.cast(nH, tf.float32)
    nW = tf.cast(nW, tf.int32)

    for b in range(nB):
        gi0 = gi[b]
        gj0 = gj[b]
        gi0f = tf.cast(gi0, tf.float32)
        gj0f = tf.cast(gj0, tf.float32)

        pred_box = pred_corners[b , gj0 * nW + gi0]
        current_gt = gt_box[b]
        # print(current_gt)
        # print(pred_box)
        conf = corner_confidence9(current_gt[1:], pred_box)
        # print(conf)
        coord_mask = tf.tensor_scatter_nd_update(coord_mask, [[b * nAnchors + nW * gj0 + gi0]],tf.constant([1.0]))
        cls_mask = tf.tensor_scatter_nd_update(cls_mask, [[b * nAnchors + nW * gj0 + gi0]],tf.constant([1.0]))
        conf_mask = tf.tensor_scatter_nd_update(conf_mask, [[b * nAnchors + nW * gj0 + gi0]],tf.cast(tf.constant([object_scale]), tf.float32))

        tx0 = tf.tensor_scatter_nd_update(tx0, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[1] * nw - gi0f])
        ty0 = tf.tensor_scatter_nd_update(ty0, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[2] * nh - gj0f])
        tx1 = tf.tensor_scatter_nd_update(tx1, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[3] * nw - gi0f])
        ty1 = tf.tensor_scatter_nd_update(ty1, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[4] * nh - gj0f])
        tx2 = tf.tensor_scatter_nd_update(tx2, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[5] * nw - gi0f])
        ty2 = tf.tensor_scatter_nd_update(ty2, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[6] * nh - gj0f])
        tx3 = tf.tensor_scatter_nd_update(tx3, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[7] * nw - gi0f])
        ty3 = tf.tensor_scatter_nd_update(ty3, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[8] * nh - gj0f])
        tx4 = tf.tensor_scatter_nd_update(tx4, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[9] * nw - gi0f])
        ty4 = tf.tensor_scatter_nd_update(ty4, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[10] * nh - gj0f])
        tx5 = tf.tensor_scatter_nd_update(tx5, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[11] * nw - gi0f])
        ty5 = tf.tensor_scatter_nd_update(ty5, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[12] * nh - gj0f])
        tx6 = tf.tensor_scatter_nd_update(tx6, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[13] * nw - gi0f])
        ty6 = tf.tensor_scatter_nd_update(ty6, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[14] * nh - gj0f])
        tx7 = tf.tensor_scatter_nd_update(tx7, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[15] * nw - gi0f])
        ty7 = tf.tensor_scatter_nd_update(ty7, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[16] * nh - gj0f])
        tx8 = tf.tensor_scatter_nd_update(tx8, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[17] * nw - gi0f])
        ty8 = tf.tensor_scatter_nd_update(ty8, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[18] * nh - gj0f])

        tconf = tf.tensor_scatter_nd_update(tconf, [[b * nAnchors + nW * gj0 + gi0]], [conf])
        tcls = tf.tensor_scatter_nd_update(tcls, [[b * nAnchors + nW * gj0 + gi0]], [current_gt[0]])

        nCorrect = tf.cond(conf > 0.5, lambda: tf.add(nCorrect, 1), lambda: tf.add(nCorrect, 0))

    coord_mask = tf.reshape(coord_mask, [nB , nH , nW])
    cls_mask = tf.reshape(cls_mask, [nB, nH, nW])
    tconf = tf.reshape(tconf, [nB , nH , nW])
    tcls = tf.reshape(tcls, [nB, nH, nW])
    conf_mask = tf.reshape(conf_mask, [nB, nH, nW])
    tx0 = tf.reshape(tx0, [nB, nH, nW])
    ty0 = tf.reshape(ty0, [nB, nH, nW])
    tx1 = tf.reshape(tx1, [nB, nH, nW])
    ty1 = tf.reshape(ty1, [nB, nH, nW])
    tx2 = tf.reshape(tx2, [nB, nH, nW])
    ty2 = tf.reshape(ty2, [nB, nH, nW])
    tx3 = tf.reshape(tx3, [nB, nH, nW])
    ty3 = tf.reshape(ty3, [nB, nH, nW])
    tx4 = tf.reshape(tx4, [nB, nH, nW])
    ty4 = tf.reshape(ty4, [nB, nH, nW])
    tx5 = tf.reshape(tx5, [nB, nH, nW])
    ty5 = tf.reshape(ty5, [nB, nH, nW])
    tx6 = tf.reshape(tx6, [nB, nH, nW])
    ty6 = tf.reshape(ty6, [nB, nH, nW])
    tx7 = tf.reshape(tx7, [nB, nH, nW])
    ty7 = tf.reshape(ty7, [nB, nH, nW])
    tx8 = tf.reshape(tx8, [nB, nH, nW])
    ty8 = tf.reshape(ty8, [nB, nH, nW])

    return  nCorrect, coord_mask, conf_mask, cls_mask, tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, ty0, ty1, ty2, ty3, ty4, ty5, ty6, ty7, ty8, tconf, tcls


class RegionLoss():
    def __init__(self, batch_size, num_classes=1):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.coord_scale = 1
        self.noobject_scale = 0.1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6

    def region_loss(self, output, target):
        # Parameters
        #Shape of prediction [ b, 13, 13, 20]

        nB = output.shape[0]
        nH = output.shape[1]
        nW = output.shape[2]
        output = tf.transpose(output, [0, 3, 2, 1])

        x0 = tf.sigmoid(output[:,0,...])
        y0 = tf.sigmoid(output[:,1,...])
        x1 = output[:,2,...]
        y1 = output[:,3,...]
        x2 = output[:,4,...]
        y2 = output[:,5,...]
        x3 = output[:,6,...]
        y3 = output[:,7,...]
        x4 = output[:,8,...]
        y4 = output[:,9,...]
        x5 = output[:,10,...]
        y5 = output[:,11,...]
        x6 = output[:,12,...]
        y6 = output[:,13,...]
        x7 = output[:,14,...]
        y7 = output[:,15,...]
        x8 = output[:,16,...]
        y8 = output[:,17,...]
        conf = tf.sigmoid(output[:,18,...])

        grid_x = tf.range(nH, dtype=tf.int32)
        grid_y = tf.range(nW, dtype=tf.int32)


        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)

        grid_x = tf.cast(grid_x, tf.float32)
        grid_y = tf.cast(grid_y, tf.float32)

        x = []
        x.append(tf.reshape(((x0 + grid_x) / tf.cast(nW, tf.float32)),[nB, nH * nW]))
        x.append(tf.reshape(((y0 + grid_y) / tf.cast(nH, tf.float32)),[nB , nH * nW]))
        x.append(tf.reshape(((x1 + grid_x) / tf.cast(nW, tf.float32)),[nB,  nH * nW]))
        x.append(tf.reshape(((y1 + grid_y) / tf.cast(nH, tf.float32)),[nB , nH * nW]))
        x.append(tf.reshape(((x2 + grid_x) / tf.cast(nW, tf.float32)),[nB , nH * nW]))
        x.append(tf.reshape(((y2 + grid_y) / tf.cast(nH, tf.float32)),[nB , nH * nW]))
        x.append(tf.reshape(((x3 + grid_x) / tf.cast(nW, tf.float32)),[nB , nH * nW]))
        x.append(tf.reshape(((y3 + grid_y) / tf.cast(nH, tf.float32)),[nB,  nH * nW]))
        x.append(tf.reshape(((x4 + grid_x) / tf.cast(nW, tf.float32)),[nB , nH * nW]))
        x.append(tf.reshape(((y4 + grid_y) / tf.cast(nH, tf.float32)),[nB  , nH * nW]))
        x.append(tf.reshape(((x5 + grid_x) / tf.cast(nW, tf.float32)),[nB  , nH * nW]))
        x.append(tf.reshape(((y5 + grid_y) / tf.cast(nH, tf.float32)),[nB  , nH * nW]))
        x.append(tf.reshape(((x6 + grid_x) / tf.cast(nW, tf.float32)),[nB  , nH * nW]))
        x.append(tf.reshape(((y6 + grid_y) / tf.cast(nH, tf.float32)),[nB  , nH * nW]))
        x.append(tf.reshape(((x7 + grid_x) / tf.cast(nW, tf.float32)),[nB  , nH * nW]))
        x.append(tf.reshape(((y7 + grid_y) / tf.cast(nH, tf.float32)),[nB  , nH * nW]))
        x.append(tf.reshape(((x8 + grid_x) / tf.cast(nW, tf.float32)),[nB  , nH * nW]))
        x.append(tf.reshape(((y8 + grid_y) / tf.cast(nH, tf.float32)),[nB  , nH * nW]))

        pred_corners = tf.stack(x)
        pred_corners = tf.transpose(pred_corners, [1,0, 2])

        nCorrect, coord_mask, conf_mask, cls_mask, tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, ty0, ty1, ty2, ty3, ty4, ty5, ty6, ty7, ty8, tconf, tcls = \
            build_targets(pred_corners, target, nB, nH,  nW, self.noobject_scale,self.object_scale, self.thresh)

        nP = conf > 0.25
        nProposals = tf.reduce_sum(tf.cast(nP, tf.float32))
        conf_mask = tf.sqrt(conf_mask)

        loss_x0 = tf.reduce_sum(tf.square(x0 - tx0) * coord_mask)/2.0
        loss_y0 = tf.reduce_sum(tf.square(y0 - ty0) * coord_mask)/2.0
        loss_x1 = tf.reduce_sum(tf.square(x1 - tx1) * coord_mask)/2.0
        loss_y1 = tf.reduce_sum(tf.square(y1 - ty1) * coord_mask)/2.0
        loss_x2 = tf.reduce_sum(tf.square(x2 - tx2) * coord_mask)/2.0
        loss_y2 = tf.reduce_sum(tf.square(y2 - ty2) * coord_mask)/2.0
        loss_x3 = tf.reduce_sum(tf.square(x3 - tx3) * coord_mask)/2.0
        loss_y3 = tf.reduce_sum(tf.square(y3 - ty3) * coord_mask)/2.0
        loss_x4 = tf.reduce_sum(tf.square(x4 - tx4) * coord_mask)/2.0
        loss_y4 = tf.reduce_sum(tf.square(y4 - ty4) * coord_mask)/2.0
        loss_x5 = tf.reduce_sum(tf.square(x5 - tx5) * coord_mask)/2.0
        loss_y5 = tf.reduce_sum(tf.square(y5 - ty5) * coord_mask)/2.0
        loss_x6 = tf.reduce_sum(tf.square(x6 - tx6) * coord_mask)/2.0
        loss_y6 = tf.reduce_sum(tf.square(y6 - ty6) * coord_mask)/2.0
        loss_x7 = tf.reduce_sum(tf.square(x7 - tx7) * coord_mask)/2.0
        loss_y7 = tf.reduce_sum(tf.square(y7 - ty7) * coord_mask)/2.0
        loss_x8 = tf.reduce_sum(tf.square(x8 - tx8) * coord_mask)/2.0
        loss_y8 = tf.reduce_sum(tf.square(y8 - ty8) * coord_mask)/2.0

        loss_conf = tf.reduce_sum(tf.square(conf - tconf) * conf_mask)/2.0

        loss_x = loss_x0 + loss_x1 + loss_x2 + loss_x3 + loss_x4 + loss_x5 + loss_x6 + loss_x7 + loss_x8
        loss_y = loss_y0 + loss_y1 + loss_y2 + loss_y3 + loss_y4  + loss_y5 + loss_y6 +loss_y7 + loss_y8

        return nCorrect, nProposals, loss_x, loss_y, loss_conf

    def compute_loss(self, region_preds, slabels):
        nCorrect, nProposals, loss_x, loss_y, loss_conf, loss = 0, 0, 0, 0, 0, 0
        # print(region_preds)
        for i in range(len(region_preds)):
            # print(i)
            pred = tf.reshape(region_preds[i],[self.batch_size, 2**i * 13, 2**i * 13, 20])
            total_loss = self.region_loss(pred, slabels)
            nCorrect += total_loss[0]
            nProposals += total_loss[1]
            loss_x += total_loss[2]
            loss_y += total_loss[3]
            loss_conf += total_loss[4]

        loss = loss_x + loss_y + loss_conf
        return [loss, loss_x, loss_y, loss_conf, nProposals, nCorrect]


def predict(output, conf_thresh, num_classes, only_objectness=1, validation=False):
    # Parameters
    batch = output.shape[0]
    h = output.shape[1]
    w = output.shape[2]

    # Activation
    all_boxes = []
    max_conf = -100000
    output = output.view(batch * anchor_dim, 19 + num_classes, h * w).transpose(0, 1).contiguous().view(
        19 + num_classes, batch * anchor_dim * h * w)
    grid_x = torch.linspace(0, w - 1, w).repeat(h, 1).repeat(batch * anchor_dim, 1, 1).view(
        batch * anchor_dim * h * w).cuda()
    grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch * anchor_dim, 1, 1).view(
        batch * anchor_dim * h * w).cuda()
    xs0 = torch.sigmoid(output[0]) + grid_x
    ys0 = torch.sigmoid(output[1]) + grid_y
    xs1 = output[2] + grid_x
    ys1 = output[3] + grid_y
    xs2 = output[4] + grid_x
    ys2 = output[5] + grid_y
    xs3 = output[6] + grid_x
    ys3 = output[7] + grid_y
    xs4 = output[8] + grid_x
    ys4 = output[9] + grid_y
    xs5 = output[10] + grid_x
    ys5 = output[11] + grid_y
    xs6 = output[12] + grid_x
    ys6 = output[13] + grid_y
    xs7 = output[14] + grid_x
    ys7 = output[15] + grid_y
    xs8 = output[16] + grid_x
    ys8 = output[17] + grid_y
    det_confs = torch.sigmoid(output[18])
    cls_confs = torch.nn.Softmax()(Variable(output[19:19 + num_classes].transpose(0, 1))).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    t1 = time.time()

    # GPU to CPU
    sz_hw = h * w
    sz_hwa = sz_hw * anchor_dim
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs0 = convert2cpu(xs0)
    ys0 = convert2cpu(ys0)
    xs1 = convert2cpu(xs1)
    ys1 = convert2cpu(ys1)
    xs2 = convert2cpu(xs2)
    ys2 = convert2cpu(ys2)
    xs3 = convert2cpu(xs3)
    ys3 = convert2cpu(ys3)
    xs4 = convert2cpu(xs4)
    ys4 = convert2cpu(ys4)
    xs5 = convert2cpu(xs5)
    ys5 = convert2cpu(ys5)
    xs6 = convert2cpu(xs6)
    ys6 = convert2cpu(ys6)
    xs7 = convert2cpu(xs7)
    ys7 = convert2cpu(ys7)
    xs8 = convert2cpu(xs8)
    ys8 = convert2cpu(ys8)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    t2 = time.time()

    # Boxes filter
    for b in range(batch):
        boxes = []
        max_conf = -1
        for cy in range(h):
            for cx in range(w):
                for i in range(anchor_dim):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > max_conf:
                        max_conf = conf
                        max_ind = ind

                    if conf > conf_thresh:
                        bcx0 = xs0[ind]
                        bcy0 = ys0[ind]
                        bcx1 = xs1[ind]
                        bcy1 = ys1[ind]
                        bcx2 = xs2[ind]
                        bcy2 = ys2[ind]
                        bcx3 = xs3[ind]
                        bcy3 = ys3[ind]
                        bcx4 = xs4[ind]
                        bcy4 = ys4[ind]
                        bcx5 = xs5[ind]
                        bcy5 = ys5[ind]
                        bcx6 = xs6[ind]
                        bcy6 = ys6[ind]
                        bcx7 = xs7[ind]
                        bcy7 = ys7[ind]
                        bcx8 = xs8[ind]
                        bcy8 = ys8[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx0 / w, bcy0 / h, bcx1 / w, bcy1 / h, bcx2 / w, bcy2 / h, bcx3 / w, bcy3 / h, bcx4 / w,
                               bcy4 / h, bcx5 / w, bcy5 / h, bcx6 / w, bcy6 / h, bcx7 / w, bcy7 / h, bcx8 / w, bcy8 / h,
                               det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind] * tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
            if len(boxes) == 0:
                bcx0 = xs0[max_ind]
                bcy0 = ys0[max_ind]
                bcx1 = xs1[max_ind]
                bcy1 = ys1[max_ind]
                bcx2 = xs2[max_ind]
                bcy2 = ys2[max_ind]
                bcx3 = xs3[max_ind]
                bcy3 = ys3[max_ind]
                bcx4 = xs4[max_ind]
                bcy4 = ys4[max_ind]
                bcx5 = xs5[max_ind]
                bcy5 = ys5[max_ind]
                bcx6 = xs6[max_ind]
                bcy6 = ys6[max_ind]
                bcx7 = xs7[max_ind]
                bcy7 = ys7[max_ind]
                bcx8 = xs8[max_ind]
                bcy8 = ys8[max_ind]
                cls_max_conf = cls_max_confs[max_ind]
                cls_max_id = cls_max_ids[max_ind]
                det_conf = det_confs[max_ind]
                box = [bcx0 / w, bcy0 / h, bcx1 / w, bcy1 / h, bcx2 / w, bcy2 / h, bcx3 / w, bcy3 / h, bcx4 / w,
                       bcy4 / h, bcx5 / w, bcy5 / h, bcx6 / w, bcy6 / h, bcx7 / w, bcy7 / h, bcx8 / w, bcy8 / h,
                       det_conf, cls_max_conf, cls_max_id]
                boxes.append(box)
                all_boxes.append(boxes)
            else:
                all_boxes.append(boxes)

        all_boxes.append(boxes)

    # print(all_boxes)
    # print(len(all_boxes))

    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1 - t0))
        print('        gpu to cpu : %f' % (t2 - t1))
        print('      boxes filter : %f' % (t3 - t2))
        print('---------------------------------')
    return all_boxes
