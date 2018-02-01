# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report



---
### Writeup / README
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 34799
* The size of the validation set is ? 4410
* The size of test set is ? 12630
* The shape of a traffic sign image is ? 32 x 32 x 3
* The number of unique classes/labels in the data set is ? 43

#### 2. Include an exploratory visualization of the dataset.

The exploratory visualization of the data set can be seen in the HTML file. It is a bar chart showing how the training data looked like. It can be seen that the distribution of the training data is not even. Some labels especially at low number region have more samples (> 1750), while some others only have less than 250 labels. This may introduce some uncertainty of training model.


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I converted the images to grayscale, to make the model simpler for normalization and training. Then I normalized the images, using the mean value of 128, to make the normalization in [-1, 1] range. To double check this, I randomly picked up a picture with its corresponding label, processed it, and showed the same normalized picture again after the processing.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Pre-processing        | 32x32x1 image                                 |
| Layer1: Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24				|
| Layer2: Convolution 5x5	    |  1x1 stride, valid padding, outputs 10x10x32 |
| RELU                  |                                               |
| Max pooling	      	| 2x2 stride,  outputs 5x5x32				|
| Make a copy of Layer2          |                                               |
| Layer3: Convolution 5x5     	| 1x1 stride, valid padding, outputs 1x1x800 |
| RELU                  |                                               |
| Flatten Layer 2       | output 800                                    |
| Flatten Layer 3       | output 800                                    |
| Concatenate two layers| output 1600                                   |
| Dropout               | keep_rate = 0.65                              |
| Layer 4: Fully connected |  input 1600, output 320                    |
| RELU                  |                                               |
| Final connected       | input 320, output 43                          |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer, BATCH_SIZE = 100, EPOCHS = 60, and a learn rate of 0.0015. Also used dropout with keep rate of 0.65.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ? 0.999
* validation set accuracy of ? 0.962
* test set accuracy of ? 0.941

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
First I just used the solution based on LeNet lab, which gives me about 0.89 on validation test.
* What were some problems with the initial architecture?
I have tried to tweak some parameters, i.e., different learn rates and batch sizes, but the result did not change that much. Now I think initial architecture does not have enough layer after first convolution, so I have added it to 28x28x24 now.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting. Which parameters were tuned? How were they adjusted and why?
Based on some previous results and gone through some discussions online, I have made 3 major changes. First is to add more layers on the first convolution (mentioned above), and second is to make a copy of the layer 2, and keep doing convolution to 1x1x800 on layer 3, then flatten both and concatenate them. The third is to add dropout. After added these changes, I have made many iterations of EPOCH, BATCH_SIZE, learn_rate, and keep_rate. I found 0.65 has better result than most of the default 0.5 keep rate, and learn_rate is set to 0.0015. I felt after 50-60 of EPOCH, the validation rate accually not increases anymore so I just set to 60. For BATCH_SIZE, I have tried 80, 100, 120, 150 and picked up 100 that has the best result.
Moreover, to modify the architect, I have tried to add more layers from 1600 down to 43, but looked like the validation accuracy drops a little and this may be due to overfitting. So I cut down one layer and now use 1600 to 320 and to 43 directly.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

In order to have more insight about the prediction accuracy, I have picked up 10 pictures online. These true labels are [4, 7, 12, 14, 17, 18, 26, 30, 34, 40].

Initially I thought pics label {18, 26, 30} are a bit hard to recognize. Since they have same outside triangle shape, and due to the resolution, it will be hard to identify inside shape well. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image	(label)		        |     Prediction (label)	        					| 
|:---------------------:|:---------------------------------------------:| 
| 70km/h (4)      		| 70km/h (4)   									| 
| 100km/h (7)    		| 60km/h (3) [incorrect] 						|
| priority road (12)	| priority road (12)						    |
| stop (14)	      		| stop (14)						 				|
| no entry (17)			| no entry (17)	      							|
| general caution (18)  | general caution (18)                          |
| traffic signals (26)  | traffic signals (26)                          |
| be aware of ice/snow (30) | be aware of ice/snow (30)                 |
| turn left ahead (34)  | turn left ahead (34)                          |
| roundabout (40) | End of no passing by vehicles over 3.5 metric tons (42) [incorrect]|


The model was able to correctly guess 8 of the 10 traffic signs, which gives an accuracy of 80%. This is relatively lower than the test set result.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

In the notebook, I have shown each picture with top 5 possibilities. This can be seen as in the picture or the output of the code, or directly in the HTML file. 

Only input sign (30) has top guess less than 100%, guessing as sign 30 (correct) with 72%, and guessing as sign 11 with 28%. Other results, even the prediction is wrong, have very strong certainty of the top guess.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

