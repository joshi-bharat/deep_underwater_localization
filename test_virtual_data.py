import  tensorflow as tf
import numpy as np
from region_loss import RegionLoss

import cv2

def get_bbox_mask(boxes, img_size , img=None):

    # print(boxes)
    x1 = boxes[0, 0]
    y1 = boxes[0, 1]
    x2 = boxes[0, 2]
    y2 = boxes[0, 3]

    # Just to check if the bounding box is correct
    if(img is not None):
        img = cv2.rectangle(img, (x1,y1), (x2,y2), color=(0,9,255), thickness=2)
        # cv2.namedWindow("Image")
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # [13, 13, 3, 5+num_class+1] `5` means coords and labels. `1` means mix up weight.
    y_true_4 = np.zeros((img_size[1] // 104, img_size[0] // 104), np.float32)


    scale_4_x1 = int(x1 // 104)
    scale_4_y1 = int(y1 // 104)
    scale_4_x2 = int(x2 // 104)
    scale_4_y2 = int(y2 // 104)
    y_true_4[int(scale_4_y1):int(scale_4_y2)+1, int(scale_4_x1):int(scale_4_x2)+1 ] = 1

    return y_true_4

tf.enable_eager_execution()

target = np.array([[[0., 0.63125, 0.31375, 0.64875, 0.42875, 0.6825, 0.40875, 0.69375,
   0.38625, 0.7325, 0.36375, 0.55, 0.27375, 0.5825, 0.25625, 0.5725,
   0.20125, 0.61125, 0.18]]]).astype(np.float32)
bbox = np.array([[231.4, 85.28, 296.91998, 173.68, 1.]]).astype(np.float32)
bbox_mask = get_bbox_mask(bbox, (418,418))

bbox_mask = tf.convert_to_tensor(bbox_mask)
bbox_mask = tf.reshape(bbox_mask, [1,4,4])
target = tf.convert_to_tensor(target)
region_loss = RegionLoss(batch_size=1, num_classes=1)
output = tf.zeros([1, 4, 4, 28])
loss = region_loss.region_loss(output, target, bbox_mask)
