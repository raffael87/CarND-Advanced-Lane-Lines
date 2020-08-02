**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration2.jpg "Distorted"
[image2]: ./output_images/calibration/calibration2.jpg "Undistorted"
[image3]: ./output_images/with_distortion.jpg "WithoutCorrection"
[image4]: ./output_images/distortion_corrected.jpg "DistortionCorrected"
[image5]: ./output_images/binary_combined.jpg "DistortionCorrected"
[image6]: ./output_images/birdseye.jpg "BirdsEyeView"
[image7]: ./output_images/window.jpg "SlidingWindow"
[image8]: ./output_images/search.jpg "SearchAroundPoly"
[image9]: ./output_images/video.png "VideoImage"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation. This project is an improvement from the first lane detection algorithm.  


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera calibration happens at the beginning of the application. The camera calibration has to be done only once because the matrix and the distance will be the same for all images.
The calibration is calculated with a 9x6 chessboard which is on 20 calibration images.
First we create object_points which represent the corners of the chessboard in the 3d world.
But as the chessboard is only a printed image, it is flat, so that we will use only x,y and z will be 0.
The object points are the same for every calibration image.
The next step is to apply the OpenCV function findChessboardCorners which gives us (if successfully detected), the corners from the chessboard in the image.
Having now object and image points, we can call the OpenCV calibrateCamera function with them. The function returns then the camera calibration and distortion coefficients.
Because we have  20 calibration images, the points are accumulated and then passed to the calibrateCamera function. In order to get the undistorted image, the OpenCV function undistort is used. Calibration steps can be found in `helper_calibration.py`.
![alt text][image1]
![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]
![alt text][image4]
With the matrix and the distance from the calibration step, I call for every image (from video for example) the undistort function from OpenCv. You can clearly see at the deer road sign how the distortion is corrected.
Calibration steps can be found in `helper_calibration.py`.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

As an improvement from the last project, where only the canny operator was used, color and gradient thresholding is used.
For color threshold, the HLS color space is split up into L (luminosity) and S (saturation) channels.
The L channel is used to generate a x gradient threshold binary image with the Sobel function.
The S channel us used for the color threshold binary image.
Each transformation has its own threshold parameters, and for the gradient the kernel size can be changed.
The two colors represent the two threshold transformations.
The processed image value range is set to binary.
Threshold steps can be found in `helper_image.py`. `get_binary_img(...)`
![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `get_warped_image()` which can be found in the file `helper_image.py`. The source and destiantion points are the following:

| Source        | Destination   |
|:-------------:|:-------------:|
| 190, 720      | 200, 720      |
| 1130, 720     | 1080, 720     |
| 729, 480      | 1080, 0       |
| 550, 480      | 200, 0        |

With these images, also the horizon is removed and a nice bird eyes view image is generated.
![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This is an important step. In this step the lines are identified and a polynomial of second grade is calculated. In the method `find_lanes()` which takes the combined binary image and the left and right lane objects as parameters, two line finding algorithms are working. At the start point, the function `sliding_window()` is called. This algorithm uses a histogram trying to find the line peaks on x. The histogram sums up all set pixels on the y axis. When the left and right "line" is identified, windows are overlaid and the set pixels inside are counted.
Image 7 shows the end result for a curved line:
![alt text][image7]
This approach is not efficient. When we have identified the line, and we have fit the polynomial, we can use another algorithm in the next round. This one starts to search in the direct neighborhood of the polynomial. It is called via `search_around_poly`. The result you can see in image 8:
![alt text][image8]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In the file `helper_image.py` in function `get_curvatures_in_meter()` a polynomial is calculated in the meter space. `left_curverad = (1+((2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.abs(2*left_fit_cr[0])` here y_eval is the max y value and left_fit_cr is the polynoial array. ym_per_pix is used to tranform pixels to meters.
The position of the car in respect to the center of the left and right lane is done with `center = (left_fit_x[-1] + right_fit_x[-1]) / 2
  x_distance = xm_per_pix * ((img_shape[1] // 2) - center)`
The two x values from the lines are substracted. The next step is then to calculate the distance to the image center of x (car). In order to have meters as unit, xm_per_pix is used to transform pixel to meters.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
In the last step the image is plotted back. This is done by the inverse warp matrix which was already calculated in step 4. The code can be directly found in the pipeline `pipeline.py`
The image is taken now from the video.
![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./works.mp4)
Here's a [link to the window video](./window_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I learned quite a lot. It was really interesting. I hope I can find more time to improve the pipeline and to be able to run also the other videos.

From performance point of view I think that the pipeline needs to improve much more. Also the robustness when having different situations. Like not finding lanes, having bad lines etc. 
