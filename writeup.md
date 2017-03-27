**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car.png
[image2]: ./output_images/not_car.png
[image3]: ./output_images/HOG_example.png
[image4]: ./output_images/sliding_windows.png
[image5]: ./output_images/example1.png
[image6]: ./output_images/example2.png
[image7]: ./output_images/example3.png
[image8]: ./output_images/pipeline1.png
[video1]: ./output_images/project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `train()` function of `run_vehicle_detection.py`, lines 47 to 109.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one the `vehicle` classes:

![alt text][image1]

And here is a `non-vehicle`:

![alt text][image2]	

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters trading off between speed and accuracy. I found that use 'ALL' for the hog_channel was critical in getting high accuracy, although it does triple the # of features and increase training time. I found that no matter what parameters I chose, the time for prediction was always extremely fast. I found that the YCrCb color space increased accuracy a little bit over RGB. I did not see a significant improvement in accuracy when changing the orient, pix_per_cell, or cell_per_block so I kept them at 9, 8, and 2 which performed well with acceptable speed. 

My final HOG parameters can be found hard-coded in the `FeatureParameters` object, lines 30 to 38 of `run_vehicle_detection.py`

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

HOG features along with histogram and spatial features for car and non-car images are extracted on lines 84 and 85 of `run_vehicle_detection.py`. The pipeline for extracting features for each image is to convert the color space, then get spatial features with `bin_spatial()`, get color features with `color_hist()`, and HOG features with `get_hog_features()`. The pipeline for extracting the features starts with `extract_features()` in `feature_utils.py` which subsequently calls the other helper functions in `feature_utils.py`. 

The car and non-car feature matrices are concatenated and then all of the features are scaled with `sklearn.StandardScaler()`, lines 92 to 96 of `run_vehicle_detection.py`. Next, a vector of labels with 1 for cars and 0 for cars is constructed, line 99. The data is then shuffled and split into 80% training and 20% test data to check for overfitting, line 105. 

Finally, the `train_svc()` funciton is called on line 108. This function is can be found in `svc_utils.py`. It uses sklearn.LinearSVC to create a support vector machine object and then fits the training data.

After training, accuracy of the fit is assessed on the test data.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented primarily in the `find_cars()` function of `box_utils.py`, lines 53 to 144. This function crops and scales the image and then calculates HOG for the relevant portion of the image. Then the function slides a window around the relevant area computing the features of the current window and using the trained support vector machine to predict whether a car is in the current window. This function is called three different times at different crop factors, scalings, and xy_skips. The xy_skips determines how much to move the sliding window between iterations. 

To determine the crop, scale, and overlap of the windows I balanced between performance and accuracy. I tried to determine window sizes that provided good enough accuracy without being too small or densely packed for efficiency. I decided to crop aggresively to the relevant y value range, and to use 3 different scales. On the smallest scale, I doubled the gap between window overlap to save computation time. 

The following image shows all of the search windows used on the left. Green are the search windows, red are windows where cars were detected, and the first window is drawn in magenta so you can see the size of the window. On the right is the resulting sum of all search windows with a vehicle detected. 

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]

![alt text][image6]

![alt text][image7]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_images/project_video_out.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video using the `find_cars()` method mentioned above. From the positive detections I created a heatmap using the `add_heat()` function in `box_utils.py` lines 7 to 16. 

I also kept track of a second heatmap to essentially filter the heatmap across time. This was done in lines 81 to 91 of `process_img()` in `VehicleDetector.py`. At each frame I decremented this second heatmap by 2 at all indices and then added in the results of the heatmap from the current frame. This way each pixel not identified with vehicle was decremented by 2 but pixels found in 1 window were decremented by 1, pixels find in 2 windows were kept the same, pixels find in 3 windows were incremented by 1 and so on. Over several frames, pixels that were successively not part of a vehicle window become large negative values and pixels that were successively part of a vehicle window become large positive values. So a high value represents high confidence of a vehicle present and a large negative value is the opposite. In order to allow this confidence map to change over time, I applied saturation limits of -12 and 8. 

Next I thresholded the heatmap at 0 to make a binary image of vehicle positions, `apply_threshold()`, lines 19 to 28 of `box_utils.py`.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap and construct bounding boxes, `draw_labeled_bboxes()`, lines 31 to 49 of `box_utils.py`

Here's an example result showing the heatmap from a test image, the result of thresholding, and the bounding boxes then overlaid on the original image:

![alt text][image8]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I spent the most time in this project trying to improve efficiency without sacrificing the accuracy and smoothness of the result. I found the most time consuming computation was calculating HOG features for small sliding windows. So I spent a lot of effort fine-tuning the window sizes and crop areas. 

I found that training the support vector machine was very straightforward and worked well. 

I would say this pipeline would be most likely to fail for cars that look different from the training data. Good training data for the support vector machine is probably the most important piece to having a successful and robust algorithm. For this to work in general, more data of more car varieties and more angles would need to be included in the training data. 

