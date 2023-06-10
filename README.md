Color Detection using K-Nearest Neighbors Classifier
This repository contains code for performing color detection using the K-Nearest Neighbors (KNN) classifier. The KNN classifier is trained on a labeled dataset of color images and can predict the color of a given image.

Prerequisites
Python 3.x
OpenCV (cv2)
NumPy
scikit-learn
Installation
Clone the repository:

shell
Copy code
git clone <repository_url>
cd <repository_name>
Install the required dependencies:

shell
Copy code
pip install opencv-python numpy scikit-learn
Usage
Set up the dataset:

Organize your labeled dataset in subdirectories, where each subdirectory represents a color category.
Replace the 'path_to_dataset' variable in the code with the path to your labeled dataset.
Training the classifier:

Run the code to load and preprocess the dataset.
Train the KNN classifier on the dataset.
The accuracy of the classifier on the test set will be displayed.
Predicting color from an image:

Replace 'path_to_image' in the code with the path to the image for color detection.
Run the code to load the image and extract color features.
The predicted color of the image will be displayed.
License
This project is licensed under the MIT License.
