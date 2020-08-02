import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import helper_calibration as hpcal
import helper_image as hpimg
from line import Line
import pipeline as pp

line_left = Line()
line_right = Line()

def process_image(image):
    global line_left
    global line_right
    img = pp.pipeline(image, line_left, line_right, cal_mtx, cal_dist)
    return img

# STEP 1 camera calibration
global cal_mtx, cal_dist
cal_mtx, cal_dist = hpcal.calibrate_camera()

project_output = 'output_images/result_project_video.mp4'
clip1 = VideoFileClip("project_video.mp4")#.subclip(0,2)
project = clip1.fl_image(process_image)
project.write_videofile(project_output, audio=False)
