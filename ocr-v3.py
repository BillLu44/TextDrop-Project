import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot
import sys
import random

# Constants
NUM_LABELS = 36
EPOCHS = 30
BATCH_SIZE = 32

# Global label binarizer/encoder
binarizer = LabelBinarizer()
le = LabelEncoder()

# Load the MNIST dataset, which includes digits 0-9
def load_mnist_data():

    # Load all train/test data, then combine them into arrays
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
    data = np.vstack([trainData]) # Stack the data vertically 
    labels = np.hstack([trainLabels])   # Stack the labels horizontally

    return (data, labels, testData, testLabels)


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

    # Extract some data for testing
    (data, test_data, labels, test_labels) = train_test_split(data, labels, test_size=0.1, stratify=labels, random_state=42)

    return (data, labels, test_data, test_labels)


# Combine two datasets together, including their data and labels. More preparation of the data.
def combineData(data1, labels1, data2, labels2):
    data = np.vstack([data1, data2])
    labels = np.hstack([labels1, labels2])
    
    data = np.array(data, dtype="float32")  # Change data type to float32

    # Add a channel dimension (decrease dimension of data by 1), then scale pixel values to [0, 1] instead of [0, 255]
    data = np.expand_dims(data, axis=-1)
    data /= 255.0

    # Binarize the labels (make them 0-1)
    labels = binarizer.fit_transform(labels)

    return (data, labels)

# Create an image data generator, which will augment images from the dataset for training.
# This creates variations of images for an artificially "larger" dataset
def createImageGenerator():
    aug = ImageDataGenerator(
        rotation_range = 10,
        zoom_range = 0.05,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        shear_range = 0.15,
        horizontal_flip = False,
        fill_mode = "nearest"
    )

    return aug

# Deal with skew using class weights
def weighClasses(labels):
    classTotals = labels.sum(axis=0)
    classWeight = {}

    # Compute a class weight for each label to deal with skew.
    # This ensures that the training process will prioritize classes (characters) with more data
    for i in range(0, len(classTotals)):
        classWeight[i] = classTotals.max() / classTotals[i]

    return classWeight


# Using tensorflow, build a CNN with various layers (convolution, max pooling, flatten, dense)
def buildModel():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), 1, activation="relu", input_shape=(28, 28, 1), kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), 1, activation="relu", kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), 1, activation="relu"))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(128, activation="relu", kernel_regularizer=l2(0.01)))
    model.add(Dense(NUM_LABELS, activation="softmax", kernel_regularizer=l2(0.01)))

    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# Train/fit the model on the data
# ADD BACK "aug"
def trainModel(model, train_data, train_labels, val_data, val_labels, class_weight):
    tensorboard_callback = TensorBoard(log_dir="logs")

    aug = createImageGenerator()
    aug.fit(train_data)

    # Calculate sample weights based on class weights
    sample_weights = np.array([class_weight[np.argmax(label)] for label in train_labels])

    # Implement early stopping
    earlyStop = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )

    # history = model.fit(aug.flow(train_data, train_labels, batch_size=BATCH_SIZE), 
    #                     validation_data=(val_data, val_labels),
    #                     # steps_per_epoch=len(train_data) // BATCH_SIZE,
    #                     epochs=EPOCHS,
    #                     verbose=1)
    history = model.fit(train_data, train_labels, batch_size=BATCH_SIZE,
                        validation_data=(val_data, val_labels),
                        epochs=EPOCHS,
                        class_weight=class_weight,
                        callbacks=[earlyStop],
                        verbose=1)

    return history


# Test model's effectiveness
def testModel(model, test_data, test_labels):
    loss, accuracy = model.evaluate(test_data, test_labels, verbose=1)
    print(f"Evaluation Loss: {loss:.4f}")
    print(f"Evaluation Accuracy: {accuracy:.4f}")

    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)

    for i in range(10):
        # rand = random.randint(0, len(test_data) - 1)
        plot.imshow(test_data[i])

        # Convert to char if applicable
        predicted_label = int(predicted_labels[i])
        if predicted_label > 9:
            predicted_label = chr(predicted_label - 9 + 64)

        plot.title(f"Label: {predicted_label}")
        plot.show()

# Save the model's trained state for later
def saveModel(model):
    model.save("OCR_model_weighted.h5")


# Output a graph of the accuracy and loss resulting from training
def plotResults(history):
    epochs = range(1, len(history["accuracy"]) + 1)

    # Accuracy
    plot.figure(figsize=(12, 5))
    plot.subplot(1, 2, 1)
    plot.plot(epochs, history["accuracy"], label="Training Accuracy")
    plot.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plot.title("Training and Validation Accuracy")
    plot.xlabel("Epochs")
    plot.xlabel("Accuracy")
    plot.legend()

    # Loss
    plot.subplot(1, 2, 2)
    plot.plot(epochs, history["loss"], label="Training Loss")
    plot.plot(epochs, history["val_loss"], label="Validation Loss")
    plot.title("Training and Validation Loss")
    plot.xlabel("Epochs")
    plot.xlabel("Loss")
    plot.legend()

    plot.show()


# Testing
if __name__ == "__main__":
    (digitData, digitLabels, testDigitData, testDigitLabels) = load_mnist_data()
    (capitalAzData, capitalAzLabels, testAzData, testAzLabels) = load_az_data("data/A_Z Handwritten Data.csv")

    # Shift the A-Z to occupy 10-35 instead, so that they don't interfere with digits 0-9
    capitalAzLabels += 10
    testAzLabels += 10

    (data, labels) = combineData(digitData, digitLabels, capitalAzData, capitalAzLabels)
    (test_data, test_labels) = combineData(testDigitData, testDigitLabels, testAzData, testAzLabels)

    (train_data, val_data, train_labels, val_labels) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)
    classWeights = weighClasses(train_labels)

    model = buildModel()
    history = trainModel(model, train_data, train_labels, val_data, val_labels, classWeights)
    testModel(model, test_data, test_labels)

    plotResults(history.history)

    saveModel(model)

