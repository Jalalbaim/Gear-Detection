# Gear-Detection
Gear detection using OpenCv and Machine Learning
## Project definition

The primary objective of my project is to create a robust gear detection system using Python and the OpenCV library. This system will be capable of accurately identifying various types of gears and distinguishing between defective gears and those that operate perfectly. Leveraging the powerful computer vision capabilities of OpenCV, the system will analyze the visual characteristics of each gear, such as size, shape, and surface texture. These features will then be extracted and used to train a machine learning model.

By employing machine learning techniques, the system will learn to differentiate between gears that are functioning correctly and those that are faulty. The model will be trained on a dataset comprising both defective and non-defective gears, allowing it to learn the distinguishing patterns and characteristics associated with each class. Once the training is complete, the system will be able to accurately classify new gears based on their extracted features.

This gear detection project holds significant potential for applications in quality control, preventive maintenance, and machinery optimization. By automatically identifying faulty gears, it can help prevent unexpected breakdowns, reduce downtime, and optimize maintenance schedules. Additionally, the system can contribute to improving overall production efficiency by ensuring the use of reliable gears in industrial processes.

## Features collection
### Line detection: 
![mort_lines](https://github.com/Jalalbaim/Gear-Detection/assets/110737334/2a1b012f-c917-4e51-a126-ba4418ee4296)
### Circle Detection:
![parfait_circles](https://github.com/Jalalbaim/Gear-Detection/assets/110737334/196e3672-1d17-41c2-ba63-733c7a70aed0)

## ML 
This is basically a gear classification problem, for which we will use a classification algorithm known as Logistic Regression.
The features that can be gathered from the tests to create our dataset are as follows: the number of pixels that make up the gear, the number of pixels that make up the background, the ratio of gear pixels to background pixels, the number of lines, and the number of circles.
Given that we only have 18 images, out of which only 2 represent a functional gear, we can conclude that our dataset is unbalanced, which will affect our accuracy score.
To address this issue, I have used ## SMOTE: Synthetic Minority Oversampling Technique ##  to oversample my data and balance the dataset.
