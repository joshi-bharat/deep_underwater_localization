# coding: utf-8

from __future__ import division, print_function

import cv2
import random


def get_color_table(class_num, seed=2):
    random.seed(seed)
    color_table = {}
    for i in range(class_num):
        color_table[i] = [random.randint(0, 255) for _ in range(3)]
    return color_table


def plot_one_box(img, coord, label=None, color=None, line_thickness=None):
    '''
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.
    '''
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

def draw_demo_img(img, projectpts, color = (0, 255, 0)):

    vertices = []
    for i in range(9):
        x = projectpts[i][0]
        y = projectpts[i][1]
        coordinates = (int(x),int(y))
        vertices.append(coordinates)
        cv2.circle(img, coordinates, 1, (0, 255, 255), -1)


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
