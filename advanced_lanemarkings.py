import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime
import os
import helper_calibration as hpcal
import helper_image as hpimg
from line import Line
import pipeline as pp

line_left = Line()
line_right = Line()

def imshow(image):
    plt.imshow(image)
    plt.show()

def imshow_gray(image):
    plt.imshow(image, cmap='gray')
    plt.show()

image_dir = "output_images/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "/"
#os.makedirs(image_dir)

#image = mpimg.imread('test_images/straight_lines1.jpg')
image = mpimg.imread('test_images/test2.jpg')


# STEP 1 camera calibration
global cal_mtx, cal_dist
cal_mtx, cal_dist = hpcal.calibrate_camera()

result = pp.pipeline(image, line_left, line_right, cal_mtx, cal_dist)
imshow(result)
