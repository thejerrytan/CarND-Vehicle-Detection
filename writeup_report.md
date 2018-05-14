## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[augment_brightness_comparision]: ./output_images/augment_brightness_comparision.png
[car_visualization]: ./output_images/car_visualization.png
[color_histogram_comparision]: ./output_images/color_histogram_comparision.png
[final_detected_cars_test1]: ./output_images/final_detected_cars_test1.jpg
[final_detected_cars_test2]: ./output_images/final_detected_cars_test2.jpg
[final_detected_cars_test3]: ./output_images/final_detected_cars_test3.jpg
[final_detected_cars_test4]: ./output_images/final_detected_cars_test4.jpg
[final_detected_cars_test5]: ./output_images/final_detected_cars_test5.jpg
[final_detected_cars_test6]: ./output_images/final_detected_cars_test6.jpg
[heatmap_test1]: ./output_images/heatmap_test1.jpg
[heatmap_test2]: ./output_images/heatmap_test2.jpg
[heatmap_test3]: ./output_images/heatmap_test3.jpg
[heatmap_test4]: ./output_images/heatmap_test4.jpg
[heatmap_test5]: ./output_images/heatmap_test5.jpg
[heatmap_test6]: ./output_images/heatmap_test6.jpg
[hog_colorspace_comparision]: ./output_images/hog_colorspace_comparision.png
[notcar_visualization]: ./output_images/notcar_visualization.png
[sliding_large_windows_test1]: ./output_images/sliding_large_windows_test1.jpg
[sliding_large_windows_test2]: ./output_images/sliding_large_windows_test2.jpg
[sliding_large_windows_test3]: ./output_images/sliding_large_windows_test3.jpg
[sliding_large_windows_test4]: ./output_images/sliding_large_windows_test4.jpg
[sliding_large_windows_test5]: ./output_images/sliding_large_windows_test5.jpg
[sliding_large_windows_test6]: ./output_images/sliding_large_windows_test6.jpg
[sliding_medium_windows_test1]: ./output_images/sliding_medium_windows_test1.jpg
[sliding_medium_windows_test2]: ./output_images/sliding_medium_windows_test2.jpg
[sliding_medium_windows_test3]: ./output_images/sliding_medium_windows_test3.jpg
[sliding_medium_windows_test4]: ./output_images/sliding_medium_windows_test4.jpg
[sliding_medium_windows_test5]: ./output_images/sliding_medium_windows_test5.jpg
[sliding_medium_windows_test6]: ./output_images/sliding_medium_windows_test6.jpg
[sliding_small_windows_test1]: ./output_images/sliding_small_windows_test1.jpg
[sliding_small_windows_test2]: ./output_images/sliding_small_windows_test2.jpg
[sliding_small_windows_test3]: ./output_images/sliding_small_windows_test3.jpg
[sliding_small_windows_test4]: ./output_images/sliding_small_windows_test4.jpg
[sliding_small_windows_test5]: ./output_images/sliding_small_windows_test5.jpg
[sliding_small_windows_test6]: ./output_images/sliding_small_windows_test6.jpg
[spatial_bining_comparision]: ./output_images/spatial_bining_comparision.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the fifth code cell of the pipeline.ipynb IPython notebook (from lines 1 through 101 of the cell).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of the vehicle and non-vehicle classes.

![alt text][car_visualization]
![alt text][notcar_visualization]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example where I varied the color space -> `RGB`, `LUV`, `YCrCb`, `HLS`, `YUV` while keeping the HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` constant:

![alt text][hog_colorspace_comparision]

#### 2. Explain how you settled on your final choice of HOG parameters.

By comparing the output of the HOG for car and non-car in the different colorspaces, we can see that the greatest contrast happens at the S and V channel of the HSV colorspace, hence I decided on these parameters for HOG

```python
colorspace = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb, to be used for HOG
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM with C = 0.1 using gridSearch over kernel = (linear, rbf) and C = [0.1, 1, 10] parameters. The code is in cell 9, under the heading "Train the classifier using SVM". The best parameters found were linear kernel and C = 0.1, with a best test accuracy of about 0.99.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code is in cell 11, where I used the find_cars() code given in the lesson to compute all hog features for the entire image once and then extract the relevant patches via sub-sampling with sliding window.

I tried out different values of scale, cell_per_block and found out that scale of 1.5 and cell_per_block of 2 works best empirically.

I implemented multi-scale search with 3 window sizes - small (32x32), medium (64x64), large(96,96). I started with the original size (64x64) which also happens to be the size which we train our SVM with, halve it to arrive at small scale for cars near the horizon, doubled it but realized the resulting size is too big, hence scaled it at 1.5 to arrive at (96x96). These numbers are also divisible by 8, which allows me to keep the HOG feature vector length constant by scaling pix_per_cell in the same direction as well, keeping number of cells per block constant at 8. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

For spatial binning, I tried 8x8, 16x16 and 32x32 and realized that most of the features of a car can still be preserved at 16x16 scale. Hence, i used that for spatial_size instead of 32x32, reducing the final feature vector length greatly with no adverse impact on the classification accuracy.

![alt text][spatial_bining_comparision]

For histogram of colors, I compared the histogram for [RGB, HSV, LUV, YCrCb, HLS, YUV] colorspaces and realized that the greatest contrast can be found in RGB, YUV and YCrCb. I chose RGB in the end because there is no need for extra color conversion step.

![alt text][color_histogram_comparision]

I realized majority of the false positives were coming from shadows or dimly lit areas, hence I augmented the training data by adding random amount of brightness adjustments to the image. The code is in cell 3 of the notebook, function adjust_brightness(). Indeed, the classifier was giving less false positives after augmentation, but it took much longer in terms of training and classification time.

![alt text][augment_brightness_comparision]

Another optimization i did was to apply histogram equalization on the image. The code is in cell 4, function equalize_hist() under heading "Preprocessing". Since the SVM does not perform well on dimly lit images as the features of the image are not discernible, we can use histogram equalization to increase contrast and bring out the shapes, edges, etc. that will help the SVM make its decision. Performing this technique on the patch of image covered by the window would be most effective but that would be prohibitively slow and computationally too costly, hence I just perform histogram equalization on the patch of the image we are searching over, bounded by ystart:ystop.

Ultimately I searched on 3 scales using HSV 3-channel HOG features plus spatially binned color and histograms of color in RGB colorspace in the feature vector, which provided a nice result.  Here are some example images at large, medium, small window sizes, tried on the 6 test images provided:

![alt text][sliding_large_windows_test1]
![alt text][sliding_large_windows_test2]
![alt text][sliding_large_windows_test3]
![alt text][sliding_large_windows_test4]
![alt text][sliding_large_windows_test5]
![alt text][sliding_large_windows_test6]
![alt text][sliding_medium_windows_test1]
![alt text][sliding_medium_windows_test2]
![alt text][sliding_medium_windows_test3]
![alt text][sliding_medium_windows_test4]
![alt text][sliding_medium_windows_test5]
![alt text][sliding_medium_windows_test6]
![alt text][sliding_small_windows_test1]
![alt text][sliding_small_windows_test2]
![alt text][sliding_small_windows_test3]
![alt text][sliding_small_windows_test4]
![alt text][sliding_small_windows_test5]
![alt text][sliding_small_windows_test6]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][heatmap_test1]
![alt text][heatmap_test2]
![alt text][heatmap_test3]
![alt text][heatmap_test4]
![alt text][heatmap_test5]
![alt text][heatmap_test6]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][heatmap_test1]
![alt text][heatmap_test2]
![alt text][heatmap_test3]
![alt text][heatmap_test4]
![alt text][heatmap_test5]
![alt text][heatmap_test6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][final_detected_cars_test1]
![alt text][final_detected_cars_test2]
![alt text][final_detected_cars_test3]
![alt text][final_detected_cars_test4]
![alt text][final_detected_cars_test5]
![alt text][final_detected_cars_test6]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

I used data augmentation to help the SVM with images with low overall brightness. This proved to be quite helpful in reducing the false positives coming from shadows in the images as the SVM now has more training examples to learn from.

I used histogram equalization on local patches of images and this proved to be effective in reducing false positives from shadows. This is because after histogram equalization, the contrast between a car and non-car image would be maximized - you can imagine them as support vectors near the decision hyperplane previously but are now pushed further away from the hyperplane.

For some reason, the SVM wrongly classifies guardrails as car occasionally. To mitigate this, we can collect more training images of guardrail labelled as not-car.

Pipeline is likely to fail in vehicle overtaking situations where the vehicle is overtaking on the immediate left/right lane as the size of the vehicle may be too big for our sliding window techinque currently.

Pipeline is also likely to fail in vehicle occulsion scenarios, where 2 or more vehicles are hiding / overlapping each other from the perspective of the camera. Since they are positively identified in the same u,v coordinates in the heatmap, there is no way of telling how many vehicles are in the same bounding box.
