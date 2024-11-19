import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plot
from boundingbox import load_data

# Load the test data (blackboard)
def loadTestData():
    print("Loading test data...")
    data = load_data("test_image/TestImage2.JPG")

    # Add a channel dimension (decrease dimension of data by 1), then scale pixel values to [0, 1] instead of [0, 255]
    data = np.expand_dims(data, axis=-1)
    data /= 255.0

    return data


# Load the saved, trained model
def loadModel():
    return load_model("models/OCR_model_pure_best.h5")


# Evaluate the model's effectiveness on the test data
# Test model's effectiveness
def evalModel(model, test_data):

    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)

    for i in range(len(test_data)):
        plot.imshow(test_data[i])

        # Convert to char if applicable
        predicted_label = int(predicted_labels[i])
        if predicted_label > 9:
            predicted_label = chr(predicted_label - 9 + 64) # 65 is the ASCII for A

        plot.title(f"Prediction: {predicted_label}")
        plot.show()


# Driver code
if __name__ == "__main__":
    data = loadTestData()
    model = loadModel()
    evalModel(model, data)

