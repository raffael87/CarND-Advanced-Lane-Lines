import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def get_calibrated_image(image, mtx, dist):
    undistorted = cv2.undistort(image, mtx, dist, None, mtx)

    #im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #cv2.imwrite('output_images/with_distortion.jpg', im_rgb)
    #im_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
    #cv2.imwrite('output_images/distortion_corrected.jpg', im_rgb)
    return undistorted

def calibrate_camera():
    folder = "camera_cal/"
    images_for_calibration = os.listdir(folder)

    # prepare object points chess
    nx = 9
    ny = 6

    # prepare object points like (0,0,0), (1,0,0), (2,0,0), ..., (7,5,0)
    object_points = np.zeros((nx * ny, 3), np.float32)
    object_points[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # convert to coordinates

    # arrays to store object points and image points
    object_points_images = []   # 3D points
    image_points_images = []    # 2D points

    for file_name in images_for_calibration:
        image = cv2.imread(folder + file_name)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

        # If found, add object points, image points
        if ret == True:
            object_points_images.append(object_points)
            image_points_images.append(corners)

            # Draw and display the corners
            #img = cv2.drawChessboardCorners(image, (nx,ny), corners, ret)
            #cv2.imshow('imgage',image)
            #cv2.waitKey(500)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points_images, image_points_images, image.shape[1::-1], None, None)

    return mtx, dist
