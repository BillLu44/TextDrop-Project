import cv2
import numpy as np
from boundingbox import load_data
from processor import image_processor
from ocr_eval import load_model
from ocr_eval import evalModel

def createPDF(img_path, model_path):

    # processor.py is not working for bad test images
    # Call image preprocessing, convert to gray scale for boxbounding
    # grayScale_img = cv2.cvtColor(image_processor(img_path), cv2.COLOR_BGR2GRAY)

    # Temporary code for testing
    grayScale_img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    
    # Call image processing to gain get test data for neural network
    print("Loading test data...")
    data = load_data(grayScale_img)

    print("Calling neural network...")
    model = load_model(model_path)
    evalModel(model, data)


# Temporary static paths
img_path = "test_image/TestImage2.JPG"
model_path = "models/OCR_model_aug.h5"
createPDF(img_path, model_path)