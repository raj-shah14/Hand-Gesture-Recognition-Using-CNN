# Hand-Gesture-Recognition-Using-CNN
Hand Gesture Recognition using Convolution Neural Networks
Datacollection.py is used for collecting train data and test data. 
It takes 1000 images of each gestures and stores them as training set

## Dataset of Hand Gestures
![alt text](https://github.com/raj-shah14/Hand-Gesture-Recognition-Using-CNN/blob/master/handgest.jpg)
Different hand Gesture for different rover actions.

## Training
The CNN model is trained and saved by using Traingest.py
Meta and index file are created which are used in predictgest.py
![alt text](https://github.com/raj-shah14/Hand-Gesture-Recognition-Using-CNN/blob/master/cnnarch.jpg)

## Testing 
Predictgest.py is used for real time prediction of the gestures.
There are a total of 10 Different Gestures that are trained.
You can see the different gestures in Training set.
The Gestures corresponds to numbers 0-9. It returns the probability value of each gesture predicted.
Real time Hand Gesture Prediction
![alt text](https://github.com/raj-shah14/Hand-Gesture-Recognition-Using-CNN/blob/master/results.jpg)
