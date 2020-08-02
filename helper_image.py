import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def get_binary_img(image, thresh_color = (150, 255), thresh_grad = (20, 100), kernel_size = 3):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2] #use saturation channel
    l_channel = hls[:,:,1]

    # because of vertical lines we use x direction
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, kernel_size)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # apply thresholds and make binary image
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_grad[0]) & (scaled_sobel <= thresh_grad[1])] = 1

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh_color[0]) & (s_channel <= thresh_color[1])] = 1

    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    #im_rgb = cv2.cvtColor(combined_binary, cv2.COLOR_BGR2RGB)
    #cv2.imwrite('output_images/binary_combined.jpg', color_binary)

    return combined_binary, color_binary


def get_warped_image(image):
    # fixed source and destination values
    top = 480 # crop the sky
    bottom = image.shape[0] #start where the car is
    x_bottom_left = 0
    x_bottom_right = image.shape[1]
    x_middle = int(image.shape[1]//2)
    x_top_left = x_middle - (x_middle * 0.20)
    x_top_right = x_middle + (x_middle * 0.20)

    # source points in original image and destination points in transformed image
    src = np.float32([[x_bottom_left, bottom],[x_bottom_right, bottom],[x_top_right,top],[x_top_left,top]])
    dst = np.float32([[0, image.shape[0]], [image.shape[1], image.shape[0]], [image.shape[1],0],[0,0]])
    image_size = (image.shape[1], image.shape[0])
    # get transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
        # warp image using transformation matrix
    warped_image = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)
    #plt.imshow(warped_image)
    #plt.show()
    return warped_image, Minv


def find_lanes(binary_warped, line_left, line_right):

    debug = []

    if (line_left.detected == False) or (line_right.detected == False):
        debug.append('Blind search')
        left_fit, right_fit, window_img = sliding_window(binary_warped)
        #plt.imshow(window_img, cmap='gray')
        #plt.show()
    else:
        debug.append('Search around poly')
        left_fit, right_fit, window_img = search_around_poly(binary_warped, line_left.recent_xfitted[-1][0], line_right.recent_xfitted[-1][0])
        #cv2.imwrite('output_images/search.jpg', window_img)

    return left_fit, right_fit, debug, window_img


def sliding_window(binary_warped):
   # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    #out_img = np.dstack((color_image, color_image, color_image))
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)

    left_x_base = np.argmax(histogram[:midpoint])
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint

    # choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    left_x_current = left_x_base
    right_x_current = right_x_base

    # Create empty lists to receive left and right lane pixel indices
    lane_left_inds = []
    lane_right_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height

        win_x_left_low = left_x_current - margin
        win_x_left_high = left_x_current + margin

        win_x_right_low = right_x_current - margin
        win_x_right_high = right_x_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_x_left_low,win_y_low), (win_x_left_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_x_right_low,win_y_low), (win_x_right_high,win_y_high),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_left_low) & (nonzerox < win_x_left_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_right_low) & (nonzerox < win_x_right_high)).nonzero()[0]

        # Append these indices to the lists
        lane_left_inds.append(good_left_inds)
        lane_right_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            left_x_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            right_x_current = np.int(np.mean(nonzerox[good_right_inds]))


    lane_left_inds = np.concatenate(lane_left_inds)
    lane_right_inds = np.concatenate(lane_right_inds)

    # extract left and right line pixel positions
    left_x = nonzerox[lane_left_inds]
    left_y = nonzeroy[lane_left_inds]
    right_x = nonzerox[lane_right_inds]
    right_y = nonzeroy[lane_right_inds]

    # polynomial fit
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)


    # debug
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    out_img[left_y, left_x] = [255, 0, 0]
    out_img[right_y, right_x] = [0, 0, 255]
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return left_fit, right_fit, out_img


def sanity_check(left_fit, right_fit, line_left, line_right, img_height):
    ploty = np.linspace(0, img_height-1, img_height)
    left_fit_x = left_fit[0] * ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit_x = right_fit[0] * ploty**2 + right_fit[1]*ploty + right_fit[2]

    # checking that they have similar curvature
    # checking that they are separated by approximately the right distance horizontally
    lane_width = np.mean(right_fit_x - left_fit_x) * 3.7/700
    lane_width_var = np.var(right_fit_x - left_fit_x)

    radius_left_bad = line_left.radius_of_curvature > 1100
    radius_right_bad = line_right.radius_of_curvature > 1100 # radius has crazy values so no good check atm

    sane = True

    if (lane_width > 5.0 and lane_width < 3 or (lane_width_var > 500)): # or radius_left_bad or radius_right_bad):
        sane = False

    return sane


def get_curvatures_in_meter(left_fit, right_fit, img_shape):
    ym_per_pix = 30/720     # meters per pixel in y dimension
    xm_per_pix = 3.7/700    # meters per pixel in x dimension

    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    #ploty = np.linspace(0, 719, num=720)
    y_eval = np.max(ploty)


    left_fit_x = left_fit[0] * ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit_x = right_fit[0] * ploty**2 + right_fit[1]*ploty + right_fit[2]

     # for meter apply transformation
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fit_x * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fit_x * xm_per_pix, 2)

    # radius
    left_curverad = (1+((2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.abs(2*left_fit_cr[0])
    right_curverad = (1+((2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.abs(2*right_fit_cr[0])
    #left_curverad = (1+((2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.abs(2*left_fit[0])
    #right_curverad = (1+((2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.abs(2*right_fit[0])

    # calculate car position relative to lane markings
    center = (left_fit_x[-1] + right_fit_x[-1]) / 2
    x_distance = xm_per_pix * ((img_shape[1] // 2) - center)

    return left_curverad, right_curverad, x_distance


def search_around_poly(binary_warped, left_fit, right_fit):
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    margin = 100
    lane_left_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) &
     (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))

    lane_right_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) &
     (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    left_x = nonzerox[lane_left_inds]
    left_y = nonzeroy[lane_left_inds]

    right_x = nonzerox[lane_right_inds]
    right_y = nonzeroy[lane_right_inds]

    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    # debug
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[lane_left_inds], nonzerox[lane_left_inds]] = [255, 0, 0]
    out_img[nonzeroy[lane_right_inds], nonzerox[lane_right_inds]] = [0, 0, 255]
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # generate a polygon to illustrate the search window area
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')

    return left_fit, right_fit, result


def recast_points(left_fit, right_fit, img_height):

    ploty = np.linspace(0, img_height-1, img_height)
    left_fit_x = left_fit[0] * ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit_x = right_fit[0] * ploty**2 + right_fit[1]*ploty + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fit_x, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    return pts

def direction_gradient(img, sobel_kernel = 3, thresh = (0, np.pi/2)):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # calculate sobel for x and y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # x direction
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # y direction

    # take absoulte values for x and y gradients
    abs_sobel_x = np.abs(sobel_x)
    abs_sobel_y = np.abs(sobel_y)

    # calculate direction of gradient
    direction = np.arctan2(abs_sobel_y, abs_sobel_x)
    plt.imshow(direction)
    plt.show()
    # convert direction value image to 8 bit
    scaled_direction = direction
    #scaled_direction = np.uint8(255*direction / np.max(direction))

    # create binary image based on gradient strenght
    sbinary = np.zeros_like(scaled_direction)
    sbinary[(scaled_direction >= thresh[0]) & (scaled_direction <= thresh[1])] = 1

    return sbinary
