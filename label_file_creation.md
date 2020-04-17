### Label file creation

You could follow these steps to create labels for your custom dataset:

1. Get the 3D bounding box surrounding the 3D object model. We use the already provided 3D object model for [Aqua2 UAV](aqua_glass_removed.ply)  to get the 3D bounding box. 
If you would like to create a 3D model for a custom object, you can refer to the Section 3.5 of the following paper and the references therein: http://cmp.felk.cvut.cz/~hodanto2/data/hodan2017tless.pdf

2. Define the 8 corners of the 3D bounding box and the centroid of the 3D object model as the virtual keypoints of the object. 8 corners correspond to the `[[min_x, min_y, min_z], [min_x, min_y, max_z], [min_x, max_y, min_z], [min_x, max_y, max_z], [max_x, min_y, min_z], [max_x, min_y, max_z], [max_x, max_y, min_z], [max_x, max_y, max_z]]` positions of the 3D object model, and the centroid corresponds to the [0, 0, 0] position.

3. Project the 3D keypoints to 2D. You can use the [compute_projection](https://github.com/joshi-bharat/deep_underwater_localization/blob/master/utils/misc_utils.py#L253:L259) function that we provide to project the 3D points in 2D. You would need to know the intrinsic calibration matrix of the camera and the ground-truth rotation and translation to project the 3D points in 2D. 
Typically, obtaining ground-truth Rt transformation matrices requires a manual and intrusive annotation effort. 
For an example of how to acquire ground-truth data for 6D pose estimation, please refer to the Section 3.1 of the [paper](http://cmp.felk.cvut.cz/~hodanto2/data/hodan2017tless.pdf) describing the T-LESS dataset.

    For Synthetic Dataset, the pose was obtained through simulation using Unreal Engine Simulator.
    
    For Pool Dataset, we use used [April Tags ROS Library](http://wiki.ros.org/ar_track_alvar) to obtain relative pose between two Aquas.
    
4. There are multiple methods to label 2D detection bounding box. Please refer to literature for that.

5. Create an array consisting of the class, 2D detection box and 2D keypoint locations and write it as a line into a text file. The label file is organized in the following order. `1st number: image_index, 2nd number: absolute_image_path
 3rd number: image_width, 4th number: image_height, 5th number:label_index, 6th number: x_min from 2D bounding box,
 7th number: y_min from 2D bounding box, 8th number: x_max from 2D bounding box, 9th number: y_max from 2D bounding box,
 10th number: x1 (x-coordinate of the first corner), 11th number: y1 (y-coordinate of the first corner), ..., 24th number: x8 (x-coordinate of the eighth corner), 25th number: y8 (y-coordinate of the eighth corner)`.
 2D projection of centroid on image is optional as `26th number: x0 (x-coordinate of the centroid),  27th number: y0 (y-coordinate of the centroid)`.
 