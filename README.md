## DeepURL: Deep Pose Estimation Framework for Underwater Relative Localization
Source Code for the paper  **DeepURL: Deep Pose Estimation Framework for Underwater Relative Localization**, 
submitted to IROS 2020. [[Paper]](https://arxiv.org/abs/2003.05523)

### Introduction
We propose a real-time deep-learning approach for determining the 6D relative pose of Autonomous Underwater  Vehicles (AUV) from single image. 
Due  to  the  pro-found difficulty of collecting ground truth images with accurate 6D poses underwater, this work utilizes 
rendered images from the  Unreal  Game  Engine  simulation  for  training.  An  image translation  network  is  employed  to
bridge  the  gap  between the  rendered  and  the  real  images  producing  synthetic  images for  training.  The  proposed
method  predicts  the  6D  pose  of  an AUV  from  a  single  image  as 2D  image  keypoints  representing 8 
corners  of  the  3D  model  of  the  AUV,  and  then  the  6D pose in the camera coordinates is determined using RANSAC-based  PnP. 

![](./images/deepcl_pipeline.png)
Fig. 1: In training (red dash-dotted), the rendered images are translated to the synthetic images resembling Aqua2 swimming
in a pool or a marine environment. The synthetic images are then fed to a common encoder, which is connected to two
decoder streams: Detection Decoder (object detection) and Pose Regression Decoder (6D pose regression). Only in inference
(violet dash-double-dotted), the predicted 2D keypoint projections of 8 corners of the 3D Aqua2 model are processed and
utilized to obtain a 6D pose using the RANSAC-based PnP algorithm.