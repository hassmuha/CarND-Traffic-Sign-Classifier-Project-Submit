#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the Numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3(color image)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing how the total number of samples are distributed among each classes. Moreover we can also visualize all the training, validation and test samples distribution in the same histogram based bar chart. 

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because as our classification is basically based on traffic sign and irespective of the color space the shape is more distint factor to recognize the image. After trial with both color and grayscale and looking at the result there was no significant improvement in classification in validation set. Moreover gray scale conversion reduce the size of input layer by exactly 3. 

Here is an example of a traffic sign image before and after grayscaling. 

![alt text][image2]

As a last step, I normalized the image data because the data in input layer is distributed from 0 255 and to get numerical stability in the overall network this step is required. If we don't normalize it the hidden layer closer to input layer start influencing the network and moreover the input to the activation function for example RELU also becomes biased. keeping thr fact that all weights and biases of hidden layers are intialized by 0 mean and 0.1 sigma, I took 5 sigma as it is expected that 99.999% values of initialied weights and biases lies in this range. This correspond to shift the image from 0 to 255 to -0.5 to 0.5.

I decided to generate additional data because the more data samples covering different scenarios we have the stronger for the network to classify different scenarios. To add more data to the the data set, each image in the training dataset randomly generate clone based on either of the following 3 techniques:
1) Histogram equalization to enhance the contrast of the images so that the shape of the traffic signs also becomes prominent 
2) Zoom in to cover the sceneario of variation in sizes in validation and testing set
3) Zoom out to cover the sceneario of variation in sizes in validation and testing set

Here is an example of an original image and an histogram eualized image:

![alt text][image3]

As shown in the figure, the difference between the original grayscale image and the histogram equalized image can be easily identified and based on color enhancement now the traffic sign is easily identifiale.

Here is an example of an original image and an zoom in version of it:

![alt text][image3]

As shown in the figure, the difference between the original grayscale image and zoom in version is that the traffic sign in zoom in version almost covers the complete image. Helps to train the network to covers scenarios where picture is taken closer to the traffic sign. 

Here is an example of an original image and an histogram eualized image:

![alt text][image3]

As shown in the figure, the difference between the original grayscale image and zoom out version is that the traffic sign in zoom out version also becomes smaller. Helps to train the network to covers scenarios where picture is taken far from the traffic sign. 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, 2x2 kernel, valid padding,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride, 2x2 kernel, valid padding,  outputs 5x5x16 				|
| Flattening			| inputs 5x5x16, output = 400.       									|
| Fully connected		| input = 400, output = 120        									|
| RELU					|												|
| Dropout				|Training phase 50%, Validation and test phase 0%												|
| Fully connected		| input = 120, output = 84        									|
| RELU					|												|
| Dropout				|Training phase 50%, Validation and test phase 0%												|
| Fully connected		| input = 84, output(logits) = 43      									|
| Softmax				| input(logits) = 43, output(probabilities) = 43        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, Loss function used was the reduce mean of the cross entropy calculated between softmax and one hot encoded labels. As recomended in Lenet exercise AdamOptimizer was used to minimize the loss using backword propagation. Batch size of 128 is selected, number of epochs to be 30 while the learning rate to be 0.002.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
taking the network exactly from Lenet, without preprossing color domain, normalization 
* What were some problems with the initial architecture?
all of the problems list here which was reduced by the presence of 

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
grayscale, Normalization, augment the dataset
50% Dropout in fully connected layers

* Which parameters were tuned? How were they adjusted and why?
changing hyper parameters to see which gives best result
range defined for hyperparameters

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Important design choices for the architecture was selection of convolution layers in the begining, activation based on Relu and fully conencted layers with dropout. As we have learned always activation in each layer means the network has learned new advanced patterns based on the simple patterns learned in previous layer. Hence for the first layers the goal is to identify the simple patterns (like edges, line, orientation and shape detection) first and for which we don't really need fully connected layer with lot of weights and biases to be optimized. In convolation layer strategy was adopted to increase depth (each plane has separate kernel that means each plane activated on distint scenarios) while reducing the dimension from striding and pooling operation. As an activaiton function in each layer Relu was used to model the non-linear as well in each layer. Fully connected layers have lower dimension as output but they have lot of parameters (weights and biases) associated with it. The purpose of these layers is to train in such a way that they activates on high level features for example at the end it has connected to output layer with number of output equal to number of distant classes. Dropout on fully connected layer provide regularization, meaning next layer never rely on any specific activation to be present in previous layer and for the previous layer it becomes a must condition to adjust the parameters in such a way that reduntant information is always present to be communicated to next layer.

If a well known architecture was chosen:
* What architecture was chosen? 
Lenet architecture was chosen and as it was originally designed for character recognition and as here where the texture and shape is the distint feature to identify I believe Lenet with some modification will work best for us as well.
As seen the validation and test
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Slippery road      		| Slippery road   									| 
| Speed limit (30km/h)     			| Speed limit (30km/h) 										|
| Roundabout mandatory					| Roundabout mandatory											|
| Stop	      		| Stop					 				|
| Speed limit (50km/h)			| Speed limit (50km/h)      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the fourth image, put first here and put bar graph as well
the model is relatively sure that this is a Slippery road sign (probability of 0.9999996424), and the image does contain a Slippery road sign. The top two soft max probabilities were shown below as rest of the softmax probabilities are zero till 10 decimal places

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9999996424         			| Slippery road   									| 
| .0000004054     				| Dangerous curve to the left 										|

For the first image, the model is relatively sure that this is a Slippery road sign (probability of 0.9999996424), and the image does contain a Slippery road sign. The top two soft max probabilities were shown below as rest of the softmax probabilities are zero till 10 decimal places

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9999996424         			| Slippery road   									| 
| .0000004054     				| Dangerous curve to the left 										|



For the second image, the model is relatively sure that this is a Slippery road sign (probability of 0.9999996424), and the image does contain a Slippery road sign. The top two soft max probabilities were shown below as rest of the softmax probabilities are zero till 10 decimal places

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9999996424         			| Slippery road   									| 
| .0000004054     				| Dangerous curve to the left 										|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


