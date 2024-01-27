# Importing necessary libraries
from sklearn import svm, metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from skimage import io, transform
import numpy as np
import time
import os

# Function to collect images and labels
def collectImages(image_paths, images, labels, label):
    for image_path in image_paths:
        image = io.imread(image_path)
        images.append(image)
        labels.append(label)

# Function to define and preprocess data
def defineData():
    class_size = 1000  # Size of each class
    
    images = []       # List to store images
    labels = []       # List to store labels
    
    # Define paths for cover images
    image_paths_cover = [dataset_dir + "/Cover/" + image_path for image_path in os.listdir(dataset_dir + "/Cover/") if image_path.endswith(".jpg")][: class_size]
    collectImages(image_paths_cover, images, labels, "cover")

    # Define paths for stego images
    image_paths_stego = [dataset_dir + "/Stego/" + image_path for image_path in os.listdir(dataset_dir + "/Stego/") if image_path.endswith(".jpg")][: class_size]
    collectImages(image_paths_stego, images, labels, "stego")

    images = np.array(images)
    labels = np.array(labels)

    n_samples = len(images)

    # Convert to vector
    data = images.reshape((n_samples, -1))
    print("\nVector shape:")
    print(data.shape)

    # Normalize data
    data = data / 255.0
    
    return data, labels

# Function to create SVM classifier
def create_svc(kernel_in, data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=0, test_size=0.3, shuffle=True)
    X_train = X_train ** 2

    # Creating SVM classifier
    if kernel_in == 'linear':
        classifier = svm.SVC(kernel=kernel_in, C=10, random_state=0)
    elif kernel_in == 'poly':
        classifier = svm.SVC(kernel=kernel_in, C=10, random_state=0, degree=2, gamma='auto')
    else:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)
        classifier = svm.SVC(kernel=kernel_in, C=1000, random_state=1234, gamma='auto')

    # Fitting the SVM classifier
    model = classifier.fit(X=X_train, y=y_train)

    # Making predictions
    prediction = classifier.predict(X_test)

    # Displaying classification report and confusion matrix
    print("Classification report for linear classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_test, prediction)))

    np.set_printoptions(precision=2)
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        X_test,
        y_test,
        display_labels=["cover", "stego"],
        cmap=plt.cm.Blues,
        normalize=None,
    )
    disp.ax_.set_title("Binary classification of images: cover vs stego")

    print(disp.confusion_matrix)
    plt.show()

    return model, classifier, prediction

# Main execution
gc.collect()
dataset_dir = "/kaggle/input/alaska2-image-steganalysis"

start_time = time.time()

# Define and preprocess data
data, labels = defineData()

# Create SVM classifier
create_svc('linear', data, labels)

print(f"Total time taken: {(time.time() - start_time):.2f} seconds")
