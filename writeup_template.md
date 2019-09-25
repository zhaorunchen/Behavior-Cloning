# **Behavioral Cloning** 

## Writeup Template

Some contral operation in BASH:
1. Using cd \home\workspace\CarND-Behavioral-Cloning-P3 to change directoray
2. Using mv \home\workspace\xxx.py \home\workspace\CarND-Behavioral-Cloning-P3\ to move file from one location to another

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Write_figures/NVIDIA.jpeg "Model Visualization"
[image2]: ./Write_figures/Finalmodel.jpeg "Final Model"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

### Model Architecture
The model architecture I used is discussed [here](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) by NVIDIA. This model has architecture as shown below:

![alt text][image1]

### Adjustment of the Model
The input image has shape (60, 266, 3). To use this model, one thing we need to consider is that our training input has different shape, which is (160, 320, 3). Therefore we need to adjust the first layer to make it accept different input shape. The final model architecture is shown below:

![alt text][image2]

### Dataset
The dataset I use is the one provided by Udacity. To read in the images, OpenCV (or cv2) is used. However, the image read in by cv2 has BGR in its color format, therefore we need to convert it back to RGB for the further process.

### Correction Factor of Steering Angle for Side Cameras
In the dataset, every single steering anlge corresponds to three front facing images (mounted on left, center and right). The correction factor suggested in the lecture is 0.2 (the steering angle for left and right image). +0.2 for the left image and -0.2 for the right image.

### Augemented Dataset
Since the dataset is donminated by 0 steering angle data, we need to create augemented dataset by adding more images. To do this, we can follow the process:

0) For every single steering angle, we have three image. But since they have the same steering angle value, we can only treat them as "one image"

1) keep the center image as the same

2) for left and right image, add and subtract 0.2 from its steering angle

3) for the images in 1) and 2), flip them by multiply steering angle by -1.

4) Finally we convert "one image" to 6 images.

### Traning Set and Validation Set

Use 15% of the total data as validation set. 

#### 2. Attempts to reduce overfitting in the model

As discussed above, the model was trained and validated on augemented data sets to ensure that the model was not overfitting or underfitting. 

To make the model focus on only useful input data, we adjust the images by cropping 70 pixels from top and 25 pixels from the bottom. This will allow the imput image only contain road and corresponding bumper on two sides.

A dropout layer is used after the first fully-connectted layer, the dropout rate is 0.25.


#### 3. Model parameter tuning

* The learning rate is default one, 0.001
* Optimizer is Adam
* Loss function is MSE (Mean Squared Error is efficient for regression problem)
* batch size is 32 (16, 32, 64, 128 were tried)
* Epochs is 2 (Tried 5, but the loss converge really quick in the first 2 epochs, and remain nearly the same after second epoch)

#### 4. Appropriate training data

Udacity dataset was used


### Discussion

Tring using workspace has two issues:

1) Since the training takes relatively long time, so if you don't sit in fornt of your laptop, the workspace will quit, and you have to train it from begining

2) When using python drive.py model.h5, in the first time, it shows Keras version error. Because the model is writen in 2.2.4 version and I was training the model in version 2.0.8. Therefore I have to install the newer version and retrain the model.
