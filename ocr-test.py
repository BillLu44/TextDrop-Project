# Plan:
# - Use NIST Special Database for digits and characters
# - MNIST for digits 0-9: https://yann.lecun.com/exdb/mnist/
# - Need to get the rest of the characters. May use this for A-Z as a starting point: https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format
# https://github.com/frereit/TensorflowHandwritingRecognition/blob/master/Preprocessing%20the%20data.md - extracting from NIST manually
# Tutorial for data prep: https://medium.com/analytics-vidhya/optical-character-recognition-using-tensorflow-533061285dd3
# Tutorial for everything else: https://youtu.be/jztwpsIzEGc?si=yt1GafU04D-fvd9S

from tensorflow.keras.datasets import mnist
import numpy as np
import cv2
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from cnn import CNN

# Constants for model training
EPOCHS = 50
INIT_LR = 0.1
BS = 128

# Load the MNIST dataset, which includes digits 0-9
def load_mnist_data():

    # Load all train/test data, then combine them into arrays
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
    data = np.vstack([trainData, testData]) # Stack the data vertically
    labels = np.hstack([trainLabels, testLabels])   # Stack the labels horizontally

    return (data, labels)


# Load the Kaggle dataset, which includes capital letters A-Z
def load_az_data(datasetPath):
    data = []
    labels = []

    # Open the csv file, load all labels from the first row
    file = open(datasetPath)

    for row in file:
        row = row.split(",")

        # The label for this row is from 0-25, with 0 = A, 1 = B, 2 = C, ... 25 = Z
        labels.append(row[0])

        # The rest of the row contains pixel values. Interpret this as a 28 x 28 image
        image = np.array([int(x) for x in row[1:]], dtype="uint8")
        image = image.reshape((28, 28))
        data.append(image)

    # Convert the arrays to the correct data types 
    data = np.array(data, dtype="float32")
    labels = np.array(labels, dtype="int")

    return (data, labels)


# Combine two datasets together, including their data and labels
def combineData(data1, labels1, data2, labels2):
    data = np.vstack([data1, data2])
    labels = np.hstack([labels1, labels2])
    
    # Resize to 32x32 for Tensorflow
    data = [cv2.resize(image, (32, 32)) for image in data]
    data = np.array(data, dtype="float32")

    print(data)

    # Add a channel dimension (decrease dimension of data by 1), then scale pixel values to [0, 1] instead of [0, 255]
    data = np.expand_dims(data, axis=-1)
    data /= 255.0

    return (data, labels)


# Prepare the dataset: Deal with skew using class weights
def weighClasses(labels):

    # Binarize the labels (make them 0-1)
    binarizer = LabelBinarizer()
    labels = binarizer.fit_transform(labels)

    classTotals = labels.sum(axis=0)
    classWeight = {}

    # Compute a class weight for each label to deal with skew.
    # This ensures that the training process will prioritize classes (characters) with more data
    for i in range(0, len(classTotals)):
        classWeight[i] = classTotals.max() / classTotals[i]

    return classWeight

    
# Create an image data generator, which will augment images from the dataset for training.
# This creates variations of images for an artificially "larger" dataset
def createImageGenerator():
    return ImageDataGenerator(
        rotation_range = 10,
        zoom_range = 0.05,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        shear_range = 0.15,
        horizontal_flip = False,
        fill_mode = "nearest"
    )

# Train the ResNet CNN
def trainModel(data):
    # Optimize the data for training
    optimized = SGD(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)

    model = CNN.buildArch(32, 32, 1, len(le.classes_))


# Testing
if __name__ == "__main__":
    (digitData, digitLabels) = load_mnist_data()
    (capitalAzData, capitalAzLabels) = load_az_data("project/data/A_Z Handwritten Data.csv")

    # Shift the A-Z to occupy 10-35 instead, so that they don't interfere with digits 0-9
    capitalAzLabels += 10

    (data, labels) = combineData(digitData, digitLabels, capitalAzData, capitalAzLabels)
    classWeight = weighClasses(labels)
    
    generator = createImageGenerator()


#