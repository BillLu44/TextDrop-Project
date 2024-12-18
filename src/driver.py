import cv2
import numpy as np
import matplotlib.pyplot as plt
from boundingbox import load_data
from processor import board_extractor
from ocr_eval import loadModel, evalModel, process_img
from pdf_gen import setup_pdf, add_page, render_pdf
#from rotate_camera import start_motor
import subprocess
import threading
import requests
import os
import time

boxes_count = 0

def get_character_count():
    return boxes_count

def addPageToPDF(img_path, model_path):

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
        add_page(pdfData, Twidth, Theight)
    else :
        print(labels_count)
        print(boxes_count)

    boxes_count = 0

    #return start_motor()    # If true, continue the picture-taking loop. If false, stop it

def flask_startup():
    subprocess.run(["shell_scripts/client_script_external_network.bash"], shell=True)

if __name__ == '__main__':
    # Temporary static paths
    model_path = "models/OCR_model_aug_5.h5"
    #flask_thread = threading.Thread(target=flask_startup)
    #flask_thread.start()
    i = 0
    while (False):
        try:
            print("trying")
            x = requests.get('http://10.42.0.170:50100/take_picture', stream=True)
            if x.status_code == 200:
                print("success")
                i += 1
                img_path = os.path.join('img_dump', f"from_pi_{i}.jpg")
                with open(img_path, "wb") as file:
                    for chunk in x.iter_content(chunk_size=8192):
                        file.write(chunk)
            print("got here")
            print(img_path)
            print("skibdi")
            # img_processing = threading.Thread(createPDF(img_path, model_path))
            createPDF(img_path, model_path)
            # img_processing.start()
            print("got here?")
            # img_processing.join()
        
        except:
            # print("image fetch failed")
            # print()
            time.sleep(10)
    
    setup_pdf()
    print("\n=====================FORMATTING BOARD 1=====================")
    addPageToPDF("test_image/BEST IMAGE 1.jpg", model_path)
    print("\n=====================FORMATTING BOARD 2=====================")
    addPageToPDF("test_image/BEST IMAGE 2.jpg", model_path)
    render_pdf()
