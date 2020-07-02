# **Behavioral Cloning Project**

## Overview
In this project, there is shown how a end-to-end deep neural network can be trained on expert demonstrations in order to drive a car along a track. The test is run in a simulator.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./media/nvidia_net.PNG "nvidia end-to-end"
[image2]: ./media/final_net.png "final net"
[image3]: ./media/bs32_lr0001_e30.png "learning curves"

---
## Files Submitted & Code Quality

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Training Strategy
In order to create a good dataset, different strategies have been applied.

* **drive in both directions**: the track has been recorded driving both clockwise and counterclockwise;
* **use data from the side cameras**: the data coming from the left and right camera have been used to increase the number of samples in the dataset. This can be done applying a correction of `+-0.2` to the original steering command;
* **use a mouse to drive the car**: the use of a mouse guarantees smoot steering commands.

The obtained dataset contains 11733 pairs of images and steering command.\
This dataset has been split (line ) to have different dataset for training and validation. In particular,
* the training set contains 8213 pairs (image, steering command);
* the validation set contains 3520 pairs (image, steering command);

All data have been recorded running the simulator at its lowest configuration:
* Resolution: 640x480
* Graphics Quality: 'Fastest'

## Model Architecture
The model (line 42-67 of [model.py](model.py)) chosen for completing the task is the [NVIDIA End-To-End](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) architecture.

This is chosen particularly because this net has been used for driving a real car within the same problem formulation.

The network architecture has 11 layers and it is showed below.

![alt text][image1]

At this network have been added a preprocessing input section and two dropout layers.\
The preprocessing input section is composed by three additional layers (line 47-49 of [model.py](model.py)):
1. Lambda layer for input normalization;
```python
Lambda(lambda x: (x / 127.5) - 1, input_shape=img_input_shape)
```
2. Cropping2D layer to remove data that doesn't regard the road;
```python
Cropping2D(cropping=((50,20), (0,0))))
```
3. Lambda layer to resize the image to match the NVIDIA End-To-End input;
```python
Lambda(lambda x: tf.image.resize_images(x, (66, 200)))
```

The two dropout layer introduced to reduce overfitting are placed
1. after the first three convolutional layer;
2. after the flatten layer.\
Both layer have set to 0.5.

The final net is shown below:

![alt text][image2]

The model was trained and validated on different data sets to ensure that the model was not overfitting.

The model used an adam optimizer(line 66 of [model.py](model.py)), so the learning rate was not tuned manually.

## Data Generators
To fill the input of the net, a ***dataGenerator(...)*** function (line 70-94 of [model.py](model.py)) has been created. This function is used both for training and validation phases.
This function also performs a data augmentation randomly flipping part of the returned net inputs (line 88-92 of [model.py](model.py))

## Training Results
The model has been trained using the following hyperparameters
* epochs = 30
* batch size = 32
* learning rate = 0.0001

The loss history on training set and validation set is showed in the image below.

![alt text][image3]

From the inspection of the image it is clear that the model does:
* No underfitting, since performance gets better over epochs on training set
* No overfitting, since validation loss decreases over epochs

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

[Here](./media/output_video.mp4) there is the result of video.py

## Open Points
* **Dataset balancing** : Generalization can be better achieved by balancing the dataset with augmentation techniques aiming in representing different turning signals
