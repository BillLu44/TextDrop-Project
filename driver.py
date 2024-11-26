import cv2
import numpy as np
from boundingbox import load_data
from processor import board_extractor
from ocr_eval import loadModel
from ocr_eval import evalModel
from pdf_gen import render_pdf
#from rotate_camera import start_motor
import subprocess
import threading
import requests
import os
import time

boxes_count = 0

def get_character_count():
    return boxes_count

def createPDF(img_path, model_path):

    # processor.py is not working for bad test images
    # Call image preprocessing, convert to gray scale for boxbounding
    grayScale_img = cv2.cvtColor(board_extractor(img_path), cv2.COLOR_BGR2GRAY)

    # Temporary code for testing
    # grayScale_img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    Theight = grayScale_img.shape[0]
    Twidth = grayScale_img.shape[1]
    
    # Call image processing to gain get test data for neural network
    print("Loading test data...")
    boxes, data = load_data(grayScale_img)

    print("Calling neural network...")
    model = loadModel(model_path)
    predicted_labels = evalModel(model, data)

    labels_count = predicted_labels.shape[0]
    global boxes_count
    boxes_count = boxes.shape[0]

    if(labels_count == boxes_count):
        pdfData = np.append(np.reshape(predicted_labels, (labels_count, 1)), boxes, axis = 1)
        render_pdf(pdfData, Twidth, Theight)
        # print(pdfData)
        # createPDF(pdfData, Twidth, Theight)
    else :
        print(labels_count)
        print(boxes_count)

    #return start_motor()    # If true, continue the picture-taking loop. If false, stop it

def flask_startup():
    subprocess.run(["shell_scripts/client_script.bash"], shell=True)

if __name__ == '__main__':
    # Temporary static paths
    #img_path = "test_image/2.jpg"
    model_path = "models/OCR_model_aug_5.h5"
    flask_thread = threading.Thread(target=flask_startup)
    flask_thread.start()
    i = 0
    while (True):
        try:
            print("trying")
            x = requests.get('http://raspberrypi.local:50100/take_picture', stream=True)
            if x.status_code == 200:
                print("success")
                i += 1
                img_path = os.path.join('img_dump', f"from_pi_{i}.jpg")
                with open(img_path, "wb") as file:
                    for chunk in x.iter_content(chunk_size=8192):
                        file.write(chunk)
            print("got here")
            img_process_thread = threading.Thread(createPDF(img_path, model_path))
            img_process_thread.start()
            img_process_thread.join()
        except:
            print("image fetch failed")
            time.sleep(10)