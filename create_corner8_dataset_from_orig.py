import cv2
import numpy as np
from utils.meshply import MeshPly
from utils.misc_utils import get_3D_corners, compute_projection
from utils.plot_utils import draw_demo_img_corners

import os

k_unreal = np.array([569.416384877, 0.0, 400.0,
                     0.0, 569.797349037, 300.0,
                     0.0, 0.0, 1.0], dtype=np.float32).reshape(3,3)

height = 600
width  = 800

_FLOAT_EPS = 1e-5

def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion
    Parameters
    ----------
    q : 4 element array-like
    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*
    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows quaternions that
    have not been normalized.
    References
    ----------
    Algorithm from http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    '''
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < _FLOAT_EPS:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])

def compose_transform(trans, rot):
    transform = np.zeros((3,4))
    transform[0:3, 0:3] = rot
    transform[0:3, 3] = trans
    return transform

def test_gt():
    gt_file = open('gt.txt', 'r')
    gt_file.readline()
    lines = gt_file.readlines()
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        file = line[1]
        img = cv2.imread(file)
        bbox = [int(x) for x in line[5:9]]

        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        points = [int(x) for x in line[9: len(line)]]
        points = np.array(points).reshape(9,2)
        draw_demo_img_corners(img, points, (0, 0, 255), 9)


        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':

    test_gt()
    # mesh = MeshPly('aqua_glass_removed.ply')
    # vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    # corners3D = get_3D_corners(vertices)
    # points = np.concatenate((corners3D, np.array([0.0, 0.0, 0.0, 1.0]).reshape(4, 1)), axis=1)
    #
    # # print(points)
    #
    # output = open('/media/bjoshi/ssd-data/deepcl-data/cyclegan_synth/output_transformed.csv','r')
    # lines = output.readlines()
    #
    # count = 0
    # all_gt = open('gt.txt', 'w')
    # root_dir = '/media/bjoshi/ssd-data/deepcl-data/cyclegan_synth/images'
    # all_gt.write("id file width height class_label boxx1 boxy1 boxx2 boxy2 x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 x6 y6 "
    #              "x7 y7 x8 y8 x0 y0\n")
    # for line in lines:
    #     line = line.strip()
    #     line_arr = line.split(',')
    #     file = line_arr[0]
    #     file_name = os.path.join(root_dir, file + '.png')
    #     trans = [ float(x) for x in line_arr[1:4]]
    #     quatx, quaty, quatz, quatw = line_arr[4: 8]
    #     quat = [float(quatw), float(quatx), float(quaty), float(quatz)]
    #     rotm = quat2mat(quat)
    #     transform = compose_transform(trans, rotm)
    #
    #     points_2D = compute_projection(points, transform, k_unreal)
    #     points_2D = np.transpose(points_2D)
    #
    #     x1 = round(float(line_arr[8])/2.0)
    #     y1 = round(float(line_arr[9])/2.0)
    #     x2 = round(x1 + float(line_arr[10])/2.0)
    #     y2 = round(y1 + float(line_arr[11])/2.0)
    #     all_gt.write('%d ' % count)
    #     all_gt.write('%s ' % file_name)
    #     all_gt.write('%d %d %d %d %d %d %d' % (width, height, 0, x1, y1, x2, y2))
    #     for p in points_2D:
    #         x = p[0]
    #         y = p[1]
    #         if x < 0:
    #             x = 0
    #         if x > width:
    #             x = width
    #         if y < 0:
    #             y = 0
    #         if y > height:
    #             y = height
    #         all_gt.write(' %d %d' % (round(x), round(y)) )
    #     all_gt.write('\n')
    #     count += 1
    # all_gt.close()