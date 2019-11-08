import  tensorflow as tf
import numpy as np
from region_loss import RegionLoss
tf.enable_eager_execution()

target = np.array([0, 0.631134, 0.252321, 0.649360, 0.404504, 0.681921, 0.378961, 0.693636, 0.348723, 0.732724, 0.318275,
                   0.550372, 0.198072, 0.582864, 0.174909, 0.572461, 0.100901, 0.611471, 0.073895]).astype(np.float32)
target = tf.reshape(tf.convert_to_tensor(target), [1,19])
print(target.shape)
region_loss = RegionLoss(batch_size=1, num_classes=1)
output = tf.zeros([1, 4, 4, 20])

loss = region_loss.region_loss(output, target)
