import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from google.colab.patches import cv2_imshow
# Function to extract color features from an image
def extract_color_features(image):
    # Resize the image to a fixed size (e.g., 32x32 pixels)
    resized_image = cv2.resize(image, (32, 32))
    
    # Convert the resized image to the HSV color space
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    
    # Flatten the HSV image into a 1D array
    flattened_image = hsv_image.flatten()
    
    return flattened_image

# Function to load and preprocess the dataset
def load_dataset():
    # Load the dataset images and labels
    # Replace 'path_to_dataset' with the path to your labeled dataset
    # The dataset should be organized in subdirectories, where each subdirectory represents a color category
    dataset_path = '/content/Color_images_labelled'
    images = []
    labels = []
    color_mapping = {}
    label_counter = 0
    
    for color_dir in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, color_dir)):
            color_mapping[label_counter] = color_dir
            for image_file in os.listdir(os.path.join(dataset_path, color_dir)):
                image_path = os.path.join(dataset_path, color_dir, image_file)
                image = cv2.imread(image_path)
                if image is not None:
                    images.append(extract_color_features(image))
                    labels.append(label_counter)
            label_counter += 1
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, color_mapping

# Load and preprocess the dataset
X_train, X_test, y_train, y_test, color_mapping = load_dataset()

# Train a k-nearest neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Predict labels for the test set
y_pred = knn_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Load an image to perform color detection
# Replace 'path_to_image' with the path to your image
image_path = '/content/1429590.jpg'
image = cv2.imread(image_path)

# Extract color features from the image
features = extract_color_features(image)

# Predict the color of the image using the trained classifier
predicted_label = knn_classifier.predict([features])[0]
predicted_color = color_mapping[predicted_label]

print("Predicted color:", predicted_color)
