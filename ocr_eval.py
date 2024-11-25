import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plot
from boundingbox import load_data
import cv2

# Constants
IMAGE_SIZE = 28

# Add artificial padding to the image, since the training images are much more "zoomed out"
def padded_img(img, amount):
    h, w = img.shape
    scale = (IMAGE_SIZE - amount) / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Create a new black canvas, with the original character in the middle
    paddedImg = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype="float32")
    y_offset = (IMAGE_SIZE - img.shape[0]) // 2
    x_offset = (IMAGE_SIZE - img.shape[1]) // 2
    y_end = y_offset + img.shape[0]
    x_end = x_offset + img.shape[1]
    paddedImg[y_offset:y_end, x_offset:x_end] = img

    return paddedImg


# Turn up the contrast of the image to match training data
# amount: How much contrast should be added
# grid_size: Size of each mini-grid in the dividing step of CLAHE.
#            Smaller = finer details, greater = smoother effect
def contrast_img(img, amount, grid_size):
    # Need to convert pixel values to integers for CLAHE. First, round the decimal value.
    img = np.rint(img)
    img = np.clip(img, 0, 255)
    img = img.astype("uint8")

    clahe = cv2.createCLAHE(clipLimit=amount, tileGridSize=grid_size)
    contrasted_img = clahe.apply(img)

    # Turn it back to float32
    contrasted_img = contrasted_img.astype("float32")
    return contrasted_img


# Artificially thicken the characters to match the training data
def dilated_img(img, amount):
    kernel = np.ones((amount, amount), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


# Preprocess the test images by adding padding and turning up contrast
def process_img(img, pad_amount, contrast_amount, grid_size, dilate_amount):
    img = padded_img(img, pad_amount)
    img = contrast_img(img, contrast_amount, grid_size)
    img = dilated_img(img, dilate_amount)
    return img


# No longer needed with driver.py implemented
# # Load the test data (blackboard)
# def loadTestData():
#     print("Loading test data...")
#     data = load_data("test_image/TestImage2.JPG")
#     data = np.array([row.reshape(IMAGE_SIZE, IMAGE_SIZE) for row in data])

#     data = np.array([padded_img(img, 8) for img in data])

#     # Add a channel dimension (decrease dimension of data by 1), then scale pixel values to [0, 1] instead of [0, 255]
#     data = np.expand_dims(data, axis=-1)
#     data /= 255.0

#     return data


# Load the saved, trained model
def loadModel(model_path):
    return load_model(model_path)


# Evaluate the model's effectiveness on the test data
# Test model's effectiveness
def evalModel(model, test_data):

    test_data = np.array([process_img(img, 7, 5.0, (1, 1), 2) for img in test_data])    # 7 for smaller/more subtle writing, 8 for larger writing

    test_data = np.expand_dims(test_data, axis=-1)
    test_data /= 255.0

    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)

    # temporarily commented out
    # for i in range(len(test_data)):
    #     plot.imshow(test_data[i])

    #     # Convert to char if applicable
    #     predicted_label = int(predicted_labels[i])
    #     if predicted_label > 9:
    #         predicted_label = chr(predicted_label - 9 + 64) # 65 is the ASCII for A

    #     plot.title(f"Prediction: {predicted_label}")
    #     plot.show()

    return predicted_labels

# No longer needed with driver.py implemented
# # Driver code
# if __name__ == "__main__":
#     data = loadTestData()
#     model = loadModel()
#     evalModel(model, data)

