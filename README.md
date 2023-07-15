# Suspicious Activity Prediction based on Deep Learning

## Introduction
Around 3 million people worldwide have been victims of some crime daily. In January 2020, Surfshark published in its data report that the increased number of cameras does not correlate with the global crime index. This is because, after a point, it is not possible to have enough security personnel keeping track of a multitude of cameras 24 x 7 for the cameras actually to have an impact. This project proposes a deep learning-based approach for detecting suspicious activities live from CCTV footage. This project proposes an ensemble model based on Long-term Recurrent Convolutional Networks (LRCN) for the effective detection of suspicious activities and motionless objects in video data. The proposed ensemble model is trained on a large-scale dataset of labeled videos containing both normal and suspicious activities. 


## DataSet Description
We collected data from multiple datasets, namely the DCSASS dataset, Real Life Violence Situations Dataset, and UCF Crime Dataset. To create a comprehensive dataset for our specific project, we merged these datasets, resulting in over 800 videos that are categorized into predefined classes. These classes encompass a range of activities, including suspicious behaviors like fighting and vandalism, as well as non-suspicious activities such as walking and running. 
To detect objects like bags, handbags, and suitcases within the videos, we employed the 'YOLOv5' pre-trained model. This model has been widely recognized for its effectiveness in object detection tasks, making it a suitable choice for our project. By leveraging the capabilities of 'YOLOv5,' we aim to accurately identify and track these specific objects of interest within the video footage.
