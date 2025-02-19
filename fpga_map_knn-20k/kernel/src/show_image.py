import glob
import argparse
import os, os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pandas as pd
import csv
# 画图
from mpl_toolkits.mplot3d import Axes3D

import math
import sys


# 1. read data
file_dir = "./build/"
file_name = file_dir + "cylindrical_projection_pointcloud.txt"
f_handler = open(file_name, 'r')
file_lines = f_handler.readlines()
f_handler.close()

# 2. read and delete first line
header_info = file_lines[0].split()
print("header info: ", header_info)
del file_lines[0]
image_row = int(header_info[0])
image_column = int(header_info[1])


# 3. prepare plot data
range_image_occupy_draw = []
range_image_empty_draw = []
position_array = []
first_circle_array = []
circle_array = []

for cnt, line in enumerate(file_lines):
    line_split = [i for i in line.split()]
    this_x = line_split[0]
    this_y = line_split[1]
    this_z = line_split[2]
    this_i = line_split[3]
    this_range = line_split[4]
    this_circle = line_split[5]
    this_linecircle = line_split[6]
    this_nearcircle = line_split[7]
    this_first_circle = line_split[8]

    row_index = math.floor(cnt / image_column)
    column_index = cnt - row_index * image_column

    draw_point = (row_index, column_index)
    if(float(this_range) > 0):
        range_image_occupy_draw.append(draw_point)
    else:
        range_image_empty_draw.append(draw_point)

    # 获得位置信息以及 circle信息
    position_array.append((row_index, column_index))
    first_circle_array.append(this_first_circle)
    circle_array.append(this_circle)

print("first_circle_array[0]", first_circle_array[0])

range_image_occupy_draw = np.array(range_image_occupy_draw).reshape((len(range_image_occupy_draw),len(range_image_occupy_draw[0])))
range_image_empty_draw = np.array(range_image_empty_draw).reshape((len(range_image_empty_draw),len(range_image_empty_draw[0])))
print("occupy and empty number: ", len(range_image_occupy_draw), len(range_image_empty_draw))

position_array_draw = np.array(position_array).reshape((len(position_array),len(position_array[0])))

# first_circle_array_draw = np.array(first_circle_array).reshape(( len(first_circle_array),1) )
first_circle_array_draw = first_circle_array
circle_array_draw = circle_array


# 4. plot figure and save
tag_0 = 'occupy'
tag_1 = 'empty'
USE_FONT_SIZE = 46

fig = plt.figure(figsize=(image_column, image_row), dpi=64)     # 这块的图片尺寸以及dpi设置很奇怪。figsize=(a,b,dpi)设置图形的大小 a为图形的宽，b为图形的高，单位为英寸。dpi为设置图形每英寸的点数
plt.plot(range_image_empty_draw[:,1], range_image_empty_draw[:,0], color='dimgrey', linestyle="", marker="s", markersize=64, label=tag_0)
plt.plot(range_image_occupy_draw[:,1], range_image_occupy_draw[:,0], color='springgreen', linestyle="", marker="s", markersize=64, label=tag_1)

# # 画 first_circle 数目
# for i in range(image_row):
#     for j in range(image_column):
#         # draw_text = first_circle_array_draw[i*image_column+j]
#         # print("drawing ", i, j, draw_text)
#         plt.text(position_array_draw[i*image_column+j][1], position_array_draw[i*image_column+j][0], first_circle_array_draw[i*image_column+j], fontsize=USE_FONT_SIZE)

# 画 circle 数目
for i in range(image_row):
    for j in range(image_column):
        plt.text(position_array_draw[i*image_column+j][1], position_array_draw[i*image_column+j][0], circle_array_draw[i*image_column+j], fontsize=USE_FONT_SIZE)

plt.plot(10, 10, color='r', linestyle="", marker="s", markersize=64, label=tag_0)
plt.text(10, 10, first_circle_array_draw[10*image_column+10], fontsize=USE_FONT_SIZE)

print("draw done")

name = file_dir + "Figure_range_image"
# plt.savefig("./" + name + ".png", bbox_inches='tight', pad_inches=0.1)


# plt.savefig("./" + name + "_first_circle.pdf", bbox_inches='tight', pad_inches=1)
plt.savefig("./" + name + "_circle.pdf", bbox_inches='tight', pad_inches=1)

# plt.show()

print("save pdf done")