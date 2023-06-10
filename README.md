
# Color Detection using Machine Learning

This repository contains code for performing color detection using machine learning techniques. The code is implemented in Python and utilizes the OpenCV library for image processing and scikit-learn for training a K-Nearest Neighbors (KNN) classifier.

## Prerequisites

To run the code, you need the following prerequisites:

- Python 3.x
- OpenCV (cv2)
- NumPy
- scikit-learn

## Installation

1. Clone the repository:

   ```shell
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install the required dependencies:

   ```shell
   pip install opencv-python numpy scikit-learn
   ```

## Usage

1. Dataset Preparation:
   - Organize your labeled dataset in subdirectories, where each subdirectory represents a color category.
   - Replace the 'path_to_dataset' variable in the code with the path to your labeled dataset.

2. Training the Color Detection Model:
   - Run the code to load and preprocess the dataset.
   - Train the KNN classifier on the dataset.
   - The accuracy of the classifier on the test set will be displayed.

3. Performing Color Detection on an Image:
   - Replace 'path_to_image' in the code with the path to the image for color detection.
   - Run the code to load the image and extract color features.
   - The predicted color of the image will be displayed.

## Contributing

Contributions to this repository are always welcome. If you have any suggestions, improvements, or bug fixes, please submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

