# Suspicious Activity Prediction based on Deep Learning

## Introduction
Around 3 million people worldwide have been victims of some crime daily. In January 2020, Surfshark published in its data report that the increased number of cameras does not correlate with the global crime index. This is because, after a point, it is not possible to have enough security personnel keeping track of a multitude of cameras 24 x 7 for the cameras actually to have an impact. This project proposes a deep learning-based approach for detecting suspicious activities live from CCTV footage. This project proposes an ensemble model based on Long-term Recurrent Convolutional Networks (LRCN) for the effective detection of suspicious activities and motionless objects in video data. The proposed ensemble model is trained on a large-scale dataset of labeled videos containing both normal and suspicious activities. 


## DataSet Description
We collected data from multiple datasets, namely the DCSASS dataset, Real Life Violence Situations Dataset, and UCF Crime Dataset. To create a comprehensive dataset for our specific project, we merged these datasets, resulting in over 800 videos that are categorized into predefined classes. These classes encompass a range of activities, including suspicious behaviors like fighting and vandalism, as well as non-suspicious activities such as walking and running. 
To detect objects like bags, handbags, and suitcases within the videos, we employed the 'YOLOv5' pre-trained model. This model has been widely recognized for its effectiveness in object detection tasks, making it a suitable choice for our project. By leveraging the capabilities of 'YOLOv5,' we aim to accurately identify and track these specific objects of interest within the video footage.

## DataSet Pre-Processing
We used the OpenCV library to read and extract frames from video files. The frames are extracted from all the videos in the selected classes in the dataset. Data augmentation methods were applied, including rotation, flipping, and brightness modifications. This improves the model's capacity to deal with realistic circumstances and learn invariant properties. After that, the retrieved frames are saved in a list. The data was then normalized by dividing each pixel value by 255, which is known as "255 normalizations" or "dividing by 255." to normalize pixel values to the range [0, 1] by dividing with the maximum value, i.e. 255. This ensures that the pixel values are represented as floating-point numbers between 0 and 1, which can be more convenient for specific operations or algorithms. Next, we set the height and width of each image frame to 64 and selected 30 frames per video as the sequence length. The list of classes used for training the model is specified in the classes variable. The extracted frames, class indexes, and video file paths are then utilized for training a deep-learning model, enabling it to classify videos into specified classes. 

The dataset was then split into train and test datasets using a 75:25 split ratio. This division allows for evaluating the performance of the trained model on unseen data and estimating its generalization capabilities.

Data pre-processing processes for motionless object recognition include determining the width and height of video frames and the frames per second (fps) value. These parameters provide crucial information about the video dimensions and timings. Additionally, the threshold and duration_threshold parameters are defined to determine the sensitivity of the motionless object detection algorithm.

## LRCN Model Creation
The LRCN model was constructed using Keras, a high-level neural networks API, and TensorFlow as a backend. The VGG16 model, pre-trained on the ImageNet dataset, was used as a feature extractor for individual frames. The fully connected layers of VGG16 were removed, and the remaining convolutional layers were used as a feature extractor for each frame in the video. The TimeDistributed layer was used to apply the VGG16 model to each frame in the video sequence.

There are two convolutional layers in the model architecture, each followed by a Rectified Linear Unit (ReLU) activation function and a Flatten layer. The TimeDistributed layer's output was sent into these layers, which retrieved more abstract information from individual frames. The flattened output was then fed into an LSTM layer, which captures the temporal relationships between the frames. Finally, for the multi-class categorization of the action in the movie, a Dense layer with a softmax activation function was utilized.

To increase the model's generalization performance, we trained several LRCN models. We employed early halting to avoid the model from over-fitting. We kept an eye on the validation set's accuracy, and if it didn't increase after ten epochs, we terminated training the model and utilized the weights with the highest validation accuracy.

To combine the predictions of the multiple LRCN models, we used the majority voting ensemble technique. For each test video, we made predictions using each of the six trained models and then selected the most common prediction as the final prediction. 

![LRCN Model](Images/LRCN_Model.png)

