import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import helper_calibration as hpcal
import helper_image as hpimg
from line import Line

def imshow(image):
    plt.imshow(image)
    plt.show()

def imshow_gray(image):
    plt.imshow(image, cmap='gray')
    plt.show()

def pipeline(image, line_left, line_right, cal_mtx, cal_dist):
    # STEP 2 apply distortion to image
    image = hpcal.get_calibrated_image(image, cal_mtx, cal_dist)


    # STEP 3 apply color thresholds and gradient thresholds
    combined_binary, color_binary = hpimg.get_binary_img(image)


    # STEP 4 apply perspective transform to binary image
    warped_image, Minv = hpimg.get_warped_image(combined_binary)
    #imshow_gray(warped_image)


    # STEP 5 find left and right lane
    left_fit, right_fit, debug, window_img = hpimg.find_lanes(warped_image, line_left, line_right)

    line_left.recent_xfitted.append([left_fit])
    line_right.recent_xfitted.append([right_fit])

    line_left.current_fit = [left_fit]
    line_right.current_fit = [right_fit]

    # polynomial coefficients averaged over the last n iterations
    if ((len(line_left.recent_xfitted) > 1)): #and len(line_right.recent_xfitted) > 1)):
        line_left.best_fit = np.mean(np.array(line_left.recent_xfitted[-15:-1]), axis = 0)
        line_right.best_fit = np.mean(np.array(line_right.recent_xfitted[-15:-1]), axis = 0)
    else:
        line_left.best_fit = line_left.recent_xfitted[-1][0]
        line_right.best_fit = line_right.recent_xfitted[-1][0]

    # difference in fit coefficients between last and new fits
    line_left.diffs = left_fit - line_left.best_fit
    line_right.diffs = right_fit - line_right.best_fit


    # STEP 6 calculate curvature for both lanes
    left_curvature, right_curvature, distance_to_center = hpimg.get_curvatures_in_meter(left_fit, right_fit, image.shape)
    line_left.radius_of_curvature = left_curvature
    line_right.radius_of_curvature = right_curvature

    # STEP 7 Perform sanity checks
    sane = hpimg.sanity_check(left_fit, right_fit, line_left, line_right, image.shape[0])

    if (sane == False):
        # delete current lines and start with window search in next iteration
        del line_left.recent_xfitted[-1]
        del line_right.recent_xfitted[-1]
        line_left.detected = False
        line_right.detected = False
        left_fit = line_left.best_fit[0]
        right_fit = line_right.best_fit[0]
        debug.append('Sanity bad')
    else:
        line_left.detected = True
        line_right.detected = True
        line_left.best_fit = np.mean(np.array(line_left.recent_xfitted[-20:]), axis=0)
        line_right.best_fit = np.mean(np.array(line_right.recent_xfitted[-20:]), axis=0)
        debug.append('Sanity good')


    # STEP 8 warp detected boundaries back to original image
    pts = hpimg.recast_points(left_fit, right_fit, image.shape[0])

    # Draw the lane onto the warped blank image
    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'l: ' + str(np.round(left_curvature)), (10, 40), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(result, 'r: ' + str(np.round(right_curvature)), (1100, 40), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(result, 'dist: ' + str(np.round(distance_to_center, 2)), (640, 700), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    debug_start_y = 0
    for text in debug:
        debug_start_y += 40
        cv2.putText(result, text, (400, debug_start_y), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    return result #window_img
