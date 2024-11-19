# Plan:
# - Use NIST Special Database for digits and characters
# - MNIST for digits 0-9: https://yann.lecun.com/exdb/mnist/
# - Need to get the rest of the characters. May use this for A-Z as a starting point: https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format
# https://github.com/frereit/TensorflowHandwritingRecognition/blob/master/Preprocessing%20the%20data.md - extracting from NIST manually
# Tutorial for data prep: https://medium.com/analytics-vidhya/optical-character-recognition-using-tensorflow-533061285dd3
# Tutorial for everything else: https://youtu.be/jztwpsIzEGc?si=yt1GafU04D-fvd9S

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import cv2
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import TensorBoard
# from cnn import CNN

# Constants for model training
EPOCHS = 20
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

# Split data into training data, validation data, and test data
def partitionData(data, labels):
    train_size = int(len(data) * 0.7)
    validation_size = int(len(data) * 0.2)
    #The rest will be the test size

    train_data = []
    train_labels = []
    validation_data = []
    validation_labels = []
    test_data = []
    test_labels = []

    for i in range(0, train_size):
        train_data.append(data[i])
        train_labels.append(labels[i])
    
    for i in range(train_size, train_size + validation_size):
        validation_data.append(data[i])
        validation_labels.append(labels[i])

    for i in range(validation_size, len(data)):
        test_data.append(data[i])
        test_labels.append(labels[i])

    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


# Using tensorflow, build a CNN with various layers (convolution, max pooling, flatten, dense)
def buildModel():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), 1, activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), 1, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3, 3), 1, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile("adam", loss=tf.losses.BinaryCrossentropy(), metrics=["accuracy"])

    model.summary()
    return model

# BROKEN
def trainModel(model, data, labels, train_data, train_labels, validation_data, validation_labels, test_data, test_labels):
    tensorboard_callback = TensorBoard(log_dir="logs")
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

    trainData = np.array(trainData, dtype="float32")
    # Add a channel dimension (decrease dimension of data by 1), then scale pixel values to [0, 1] instead of [0, 255]
    trainData = np.expand_dims(trainData, axis=-1)
    trainData /= 255.0

    # Binarize the labels (make them 0-1)
    binarizer = LabelBinarizer()
    trainLabels = binarizer.fit_transform(trainLabels)

    hist = model.fit(trainData, trainLabels, epochs=EPOCHS, batch_size=1, validation_split=0.2, callbacks=[tensorboard_callback])


# Testing
if __name__ == "__main__":
    (digitData, digitLabels) = load_mnist_data()
    (capitalAzData, capitalAzLabels) = load_az_data("data/A_Z Handwritten Data.csv")

    # Shift the A-Z to occupy 10-35 instead, so that they don't interfere with digits 0-9
    capitalAzLabels += 10

    (data, labels) = combineData(digitData, digitLabels, capitalAzData, capitalAzLabels)
    classWeight = weighClasses(labels)
    
    generator = createImageGenerator()
    (train_data, train_labels, validation_data, validation_labels, test_data, test_labels) = partitionData(data, labels)
    model = buildModel()
    trainModel(model, data, labels, train_data, train_labels, validation_data, validation_labels, test_data, test_labels)

    


#