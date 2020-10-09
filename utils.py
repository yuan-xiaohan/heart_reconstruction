import cv2
import nrrd
import numpy as np
import os
import glob

# 读取文件并进行排序
def ContentSort(in_dir):
    in_content_name = os.listdir(in_dir)  # 返回目录下的所有文件和目录名
    # content_num = len(in_content_name)
    sort_num_first = []
    for file in in_content_name:
        sort_num_first.append(int((file.split("_")[2]).split(".")[0]))  # “BG_A_+序号.png” 根据 _ 分割，然后根据 . 分割，转化为数字类型
        sort_num_first.sort()
    sorted_file = []
    for sort_num in sort_num_first: #重新排序
        for file in in_content_name:
            if str(sort_num) == (file.split("_")[2]).split(".")[0]:
                sorted_file .append(file)
    return sorted_file

# Axial to Sagittal
def AxialToSagittal(a):
    s = a.transpose((2, 0, 1))
    s = s[::-1, :, :]
    return s

# Axial to Coronal
def AxialToCoronal(a):
    c = a.transpose((2,1,0))
    c = c[::-1,:,:]
    return c

# 映射到0~1之间
def Normalization(hu_value):
    hu_min = np.min(hu_value)
    hu_max = np.max(hu_value)
    normal_value = (hu_value - hu_min) / (hu_max - hu_min)
    return normal_value

# 归一化到（0，1）之间
def norm_img(image):
    if not (np.max(image) == np.min(image)):
        image_new = (image - np.min(image)) / (np.max(image)-np.min(image))
    else:
        image_new = image
    return image_new


# 给定最大最小值进行归一化
def norm_img_config(data, min, max):
    image_new = (data - min) / (max - min)
    return image_new


# 根据窗宽、窗位计算出窗的最大值和最小值
def windowAdjust(img, ww, wl):
    win_min = wl - ww / 2
    win_max = wl + ww / 2
    # 根据窗最大值、最小值来截取img
    img_new = np.clip(img, win_min, win_max)
    return img_new

# 背景顺时针90度旋转 + 水平镜像
def ImgTrasform(img):
    row, col = img.shape[:2]
    M = cv2.getRotationMatrix2D((col / 2, row / 2), -90, 1)
    img_new = cv2.flip(cv2.warpAffine(img, M, (col, row)), 1)
    return img_new

# 反变换：水平镜像 + 逆时针90度旋转
def Inverse_ImgTrasform(img):
    row, col = img.shape[:2]
    img = cv2.flip(img, 1)
    M = cv2.getRotationMatrix2D((col / 2, row / 2), 90, 1)
    img_new = cv2.warpAffine(img, M, (col, row))
    return img_new

# png to 3D mat
def png2mat(out_dir, sorted_file):
    num = len(sorted_file)
    matrix = np.zeros((512, 512, num), dtype=np.uint8)
    n = 0
    for in_name in sorted_file:
        in_content_path = os.path.join(out_dir, in_name)
        matrix[:, :, n] = cv2.imread(in_content_path)[:, :, 1]
        n = n + 1
    return matrix

# 取名字时数字对齐
def changenum(i):
    if i < 10:
        j = '000' + str(i)
    elif (i > 9 and i < 100):
        j = '00' + str(i)
    else:
        j = '0' + str(i)
    return j

# 3D mat to png
def mat2png(mat, title, out_dir):
    for index in range(mat.shape[2]):
        img = mat[:, :, index]
        cv2.imwrite(out_dir + title + changenum(index+1) + ".png", img)  # 命名为“BG_+序号.png”