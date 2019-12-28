import time
from utils import *
import tensorflow as tf
import numpy as np
from utils.misc_utils import get_bbox_mask

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

# def build_targets(pred_corners, target, nB, nH, nW, noobject_scale, object_scale, sil_thresh):
#     # nB = target.shape[0]
#
#     conf_mask = np.ones([nB, nH, nW]) * noobject_scale
#     coord_mask = tf.zeros([nB * nH * nW])
#     cls_mask = tf.zeros([nB * nH * nW])
#     tx0 = tf.zeros([nB * nH * nW])
#     ty0 = tf.zeros([nB * nH * nW])
#     tx1 = tf.zeros([nB * nH * nW])
#     ty1 = tf.zeros([nB * nH * nW])
#     tx2 = tf.zeros([nB * nH * nW])
#     ty2 = tf.zeros([nB * nH * nW])
#     tx3 = tf.zeros([nB * nH * nW])
#     ty3 = tf.zeros([nB * nH * nW])
#     tx4 = tf.zeros([nB * nH * nW])
#     ty4 = tf.zeros([nB * nH * nW])
#     tx5 = tf.zeros([nB * nH * nW])
#     ty5 = tf.zeros([nB * nH * nW])
#     tx6 = tf.zeros([nB * nH * nW])
#     ty6 = tf.zeros([nB * nH * nW])
#     tx7 = tf.zeros([nB * nH * nW])
#     ty7 = tf.zeros([nB * nH * nW])
#     tx8 = tf.zeros([nB * nH * nW])
#     ty8 = tf.zeros([nB * nH * nW])
#     tconf = tf.zeros([nB * nH * nW])
#     tcls = tf.zeros([nB * nH * nW])
#
#     nAnchors = nH * nW
#
#     cur_confs = tf.zeros([nB, nAnchors])
#
#     targets = tf.reshape(target, [-1,19])
#     targets = tf.reshape(targets[:, 1:19], [-1, 18, 1])
#     cur_gt_corners = tf.tile(targets, [1,1, nAnchors])
#     cur_confs = tf.maximum(cur_confs, corner_confidences9(cur_gt_corners, pred_corners))
#     cur_confs = tf.reshape(cur_confs, [nB, nH, nW])
#     conf_mask = conf_mask * tf.cast( cur_confs <= sil_thresh, tf.float32)
#
#     conf_mask = tf.reshape(conf_mask, [nB* nH* nW])
#
#     # print(conf_mask)
#     nCorrect = 0
#
#     gt_box = tf.reshape(target, [nB, 19])
#     gi = tf.cast(gt_box[:, 1] * tf.cast(nW, tf.float32), tf.int32)
#     gj = tf.cast(gt_box[:, 2] *  tf.cast(nH, tf.float32), tf.int32)
#
#     pred_corners = tf.transpose(pred_corners, [0, 2, 1]) # make [b,169, 18]
#
#     nw = tf.cast(nW, tf.float32)
#     nh = tf.cast(nH, tf.float32)
#     nW = tf.cast(nW, tf.int32)
#
#     for b in range(nB):
#         gi0 = gi[b]
#         gj0 = gj[b]
#         gi0f = tf.cast(gi0, tf.float32)
#         gj0f = tf.cast(gj0, tf.float32)
#
#         pred_box = pred_corners[b , gj0 * nW + gi0]
#         current_gt = gt_box[b]
#         # print(current_gt)
#         # print(pred_box)
#         conf = corner_confidence9(current_gt[1:], pred_box)
#         # print(conf)
#         coord_mask = tf.tensor_scatter_nd_update(coord_mask, [[b * nAnchors + nW * gj0 + gi0]],tf.constant([1.0]))
#         cls_mask = tf.tensor_scatter_nd_update(cls_mask, [[b * nAnchors + nW * gj0 + gi0]],tf.constant([1.0]))
#         conf_mask = tf.tensor_scatter_nd_update(conf_mask, [[b * nAnchors + nW * gj0 + gi0]],tf.cast(tf.constant([object_scale]), tf.float32))
#
#         tx0 = tf.tensor_scatter_nd_update(tx0, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[1] * nw - gi0f])
#         ty0 = tf.tensor_scatter_nd_update(ty0, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[2] * nh - gj0f])
#         tx1 = tf.tensor_scatter_nd_update(tx1, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[3] * nw - gi0f])
#         ty1 = tf.tensor_scatter_nd_update(ty1, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[4] * nh - gj0f])
#         tx2 = tf.tensor_scatter_nd_update(tx2, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[5] * nw - gi0f])
#         ty2 = tf.tensor_scatter_nd_update(ty2, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[6] * nh - gj0f])
#         tx3 = tf.tensor_scatter_nd_update(tx3, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[7] * nw - gi0f])
#         ty3 = tf.tensor_scatter_nd_update(ty3, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[8] * nh - gj0f])
#         tx4 = tf.tensor_scatter_nd_update(tx4, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[9] * nw - gi0f])
#         ty4 = tf.tensor_scatter_nd_update(ty4, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[10] * nh - gj0f])
#         tx5 = tf.tensor_scatter_nd_update(tx5, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[11] * nw - gi0f])
#         ty5 = tf.tensor_scatter_nd_update(ty5, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[12] * nh - gj0f])
#         tx6 = tf.tensor_scatter_nd_update(tx6, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[13] * nw - gi0f])
#         ty6 = tf.tensor_scatter_nd_update(ty6, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[14] * nh - gj0f])
#         tx7 = tf.tensor_scatter_nd_update(tx7, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[15] * nw - gi0f])
#         ty7 = tf.tensor_scatter_nd_update(ty7, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[16] * nh - gj0f])
#         tx8 = tf.tensor_scatter_nd_update(tx8, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[17] * nw - gi0f])
#         ty8 = tf.tensor_scatter_nd_update(ty8, [[b * nAnchors + nW * gj0 + gi0]],[current_gt[18] * nh - gj0f])
#
#         tconf = tf.tensor_scatter_nd_update(tconf, [[b * nAnchors + nW * gj0 + gi0]], [conf])
#         tcls = tf.tensor_scatter_nd_update(tcls, [[b * nAnchors + nW * gj0 + gi0]], [current_gt[0]])
#
#         nCorrect = tf.cond(conf > 0.5, lambda: tf.add(nCorrect, 1), lambda: tf.add(nCorrect, 0))
#
#     coord_mask = tf.reshape(coord_mask, [nB , nH , nW])
#     cls_mask = tf.reshape(cls_mask, [nB, nH, nW])
#     tconf = tf.reshape(tconf, [nB , nH , nW])
#     tcls = tf.reshape(tcls, [nB, nH, nW])
#     conf_mask = tf.reshape(conf_mask, [nB, nH, nW])
#     tx0 = tf.reshape(tx0, [nB, nH, nW])
#     ty0 = tf.reshape(ty0, [nB, nH, nW])
#     tx1 = tf.reshape(tx1, [nB, nH, nW])
#     ty1 = tf.reshape(ty1, [nB, nH, nW])
#     tx2 = tf.reshape(tx2, [nB, nH, nW])
#     ty2 = tf.reshape(ty2, [nB, nH, nW])
#     tx3 = tf.reshape(tx3, [nB, nH, nW])
#     ty3 = tf.reshape(ty3, [nB, nH, nW])
#     tx4 = tf.reshape(tx4, [nB, nH, nW])
#     ty4 = tf.reshape(ty4, [nB, nH, nW])
#     tx5 = tf.reshape(tx5, [nB, nH, nW])
#     ty5 = tf.reshape(ty5, [nB, nH, nW])
#     tx6 = tf.reshape(tx6, [nB, nH, nW])
#     ty6 = tf.reshape(ty6, [nB, nH, nW])
#     tx7 = tf.reshape(tx7, [nB, nH, nW])
#     ty7 = tf.reshape(ty7, [nB, nH, nW])
#     tx8 = tf.reshape(tx8, [nB, nH, nW])
#     ty8 = tf.reshape(ty8, [nB, nH, nW])
#
#     return  nCorrect, coord_mask, conf_mask, cls_mask, tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, ty0, ty1, ty2, ty3, ty4, ty5, ty6, ty7, ty8, tconf, tcls
class RegionLoss():
    def __init__(self, batch_size, num_classes=1):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.coord_scale = 1
        self.noobject_scale = 0.1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6

    def region_loss(self, output, target, bbox_mask):
        # Parameters
        #Shape of prediction [ b, 13, 13, 20]

        nB = output.shape[0]
        nH = output.shape[1]
        nW = output.shape[2]
        output = tf.transpose(output, [0, 3, 1, 2])

        x = output[:,0:9,...]
        y = output[:,9:18,...]
        conf = tf.sigmoid(output[:,18:27,...])

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
        for i in range(1):  #Change this later
            # print(i)
            pred = tf.reshape(region_preds[i],[self.batch_size, 2**i * 13, 2**i * 13, 28])
            total_loss = self.region_loss(pred, slabels, bbox_mask[i])
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

            conf = output[..., 18:27]

            output = tf.transpose(output, [0, 3, 1, 2])
            x = output[:, 0:9, ...]
            y = output[:, 9:18, ...]

            predx = (x + grid_x) / tf.cast(w, tf.float32)
            predy = (y + grid_y) / tf.cast(h, tf.float32)

            predx = tf.transpose(predx, [0, 2, 3, 1])
            predy = tf.transpose(predy, [0, 2, 3, 1])

            #Ignoring batch size and assuming single image
            #Need to fix later

            predx = tf.reshape(predx, [h, w, 9])
            predy = tf.reshape(predy, [h, w, 9])
            conf = tf.reshape(conf, [h, w, 9])

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

                x = tf.reshape(x, [h*w, 9])
                y = tf.reshape(y, [h*w, 9])
                conf = tf.reshape(conf, [h * w, 9])
                x_list.append(x)
                y_list.append(y)
                confs_list.append(conf)

        # collect results on three scales
        # take 416*416 input image for example:
        # shape: [inside_masks, 9]
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
        selected = tf.linalg.norm(center_xy - ref_xy, axis=1) < 0.2

        return pred_x, pred_y, pred_conf, selected

    def predict_one(self, output, num_classes):

        # Parameters
        batch = output.shape[0]
        h = output.shape[1]
        w = output.shape[2]

        # use some broadcast tricks to get the mesh coordinates
        grid_x = tf.range(h, dtype=tf.int32)
        grid_y = tf.range(w, dtype=tf.int32)

        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        # print(grid_x)
        # print(grid_y)
        x_offset = tf.reshape(grid_x, (-1, 1))
        y_offset = tf.reshape(grid_y, (-1, 1))

        # print(x_offset)
        # print(y_offset)

        output = output[0]

        x_offset = tf.cast(x_offset, tf.float32)
        y_offset = tf.cast(y_offset, tf.float32)


        xs0 = tf.reshape(tf.sigmoid(output[:, :, 0]), [h*w, 1]) + x_offset
        ys0 =  tf.reshape(tf.sigmoid(output[:, :, 1]), [h*w, 1]) + y_offset
        xs1 = tf.reshape(output[:, :, 2], [h * w, 1]) + x_offset
        ys1 = tf.reshape(output[:, :, 3], [h * w, 1]) + y_offset
        xs2 = tf.reshape(output[:, :, 4], [h * w, 1]) + x_offset
        ys2 = tf.reshape(output[:, :, 5], [h * w, 1]) + y_offset
        xs3 = tf.reshape(output[:, :, 6], [h * w, 1]) + x_offset
        ys3 = tf.reshape(output[:, :, 7], [h * w, 1]) + y_offset
        xs4 = tf.reshape(output[:, :, 8], [h * w, 1]) + x_offset
        ys4 = tf.reshape(output[:, :, 9], [h * w, 1]) + y_offset
        xs5 = tf.reshape(output[:, :, 10], [h * w, 1]) + x_offset
        ys5 = tf.reshape(output[:, :, 11], [h * w, 1]) + y_offset
        xs6 = tf.reshape(output[:, :, 12], [h * w, 1]) + x_offset
        ys6 = tf.reshape(output[:, :, 13], [h * w, 1]) + y_offset
        xs7 = tf.reshape(output[:, :, 14], [h * w, 1]) + x_offset
        ys7 = tf.reshape(output[:, :, 15], [h * w, 1]) + y_offset
        xs8 = tf.reshape(output[:, :, 16], [h * w, 1]) + x_offset
        ys8 = tf.reshape(output[:, :, 17], [h * w, 1]) + y_offset


        det_confs = tf.reshape(tf.sigmoid(output[:, :, 18]), [h*w,1])

        max_conf_index = tf.argmax(det_confs, 0)[0]
        # print(max_conf_index)


        bcx0 = xs0[max_conf_index, 0]
        bcy0 = ys0[max_conf_index, 0]
        bcx1 = xs1[max_conf_index, 0]
        bcy1 = ys1[max_conf_index, 0]
        bcx2 = xs2[max_conf_index, 0]
        bcy2 = ys2[max_conf_index, 0]
        bcx3 = xs3[max_conf_index, 0]
        bcy3 = ys3[max_conf_index, 0]
        bcx4 = xs4[max_conf_index, 0]
        bcy4 = ys4[max_conf_index, 0]
        bcx5 = xs5[max_conf_index, 0]
        bcy5 = ys5[max_conf_index, 0]
        bcx6 = xs6[max_conf_index, 0]
        bcy6 = ys6[max_conf_index, 0]
        bcx7 = xs7[max_conf_index, 0]
        bcy7 = ys7[max_conf_index, 0]
        bcx8 = xs8[max_conf_index, 0]
        bcy8 = ys8[max_conf_index, 0]
        max_conf = det_confs[max_conf_index, 0]

        w = tf.cast(w, tf.float32)
        h = tf.cast(h, tf.float32)

        box = [bcx0 / w, bcy0 / h, bcx1 / w, bcy1 / h, bcx2 / w, bcy2 / h, bcx3 / w, bcy3 / h, bcx4 / w,
        bcy4 / h, bcx5 / w, bcy5 / h, bcx6 / w, bcy6 / h, bcx7 / w, bcy7 / h, bcx8 / w, bcy8 / h, max_conf]


        return tf.stack(box)

    def build_targets(self, pred_x, pred_y, target, bbox_mask, grid_x, grid_y):

        nB = pred_x.shape[0]
        nH = pred_x.shape[2]
        nW = pred_x.shape[3]

        nAnchors = nH * nW

        conf_mask = tf.ones([nB, nH, nW, 9])

        targets = tf.reshape(target, [-1, 19])
        targets = targets[:,1:19]
        # print(targets)
        target_x = targets[:, ::2]
        target_y = targets[:, 1::2]

        target_x = tf.expand_dims(target_x, axis=2)
        target_y = tf.expand_dims(target_y, axis=2)

        pred_x = tf.reshape(pred_x, [-1, 9, nAnchors])
        pred_y = tf.reshape(pred_y, [-1, 9, nAnchors])

        target_x = tf.tile(target_x, [1,1,nAnchors])
        target_y = tf.tile(target_y, [1,1,nAnchors])

        cur_confs = self.corner_confidences9(pred_x, target_x, pred_y, target_y)

        cur_confs = tf.reshape(cur_confs, [nB, nH, nW, 9])

        bbox_masks = tf.expand_dims(bbox_mask, axis=3)
        bbox_masks = tf.tile(bbox_masks, [1,1,1, 9])

        conf_noobj_mask = conf_mask * self.noobject_scale * tf.cast(cur_confs <= self.thresh, tf.float32) *\
                          tf.cast(tf.logical_not(tf.cast(bbox_masks, tf.bool)), tf.float32)

        #removing the noobj mask, not sure if it will improve the results
        conf_mask = conf_mask * bbox_masks * self.object_scale
        cur_confs = cur_confs * bbox_masks

        target_x = tf.reshape(target_x, [nB, 9, nW, nH])
        target_y = tf.reshape(target_y, [nB, 9, nW, nH])

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
