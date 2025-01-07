# TextDrop - Project

TextDrop is an embedded systems project developed by Aleksa Misc, Bill Lu, Samantha Mac, Sean Chen and William Dai for the SE-101 course at the University of Waterloo. 

## Proposed Project:

By demo day, our goal was to build TextDrop, an automated note-taking device that extracts handwritten text on a blackboard and formats it into a multi-page PDF file. Our minimum viable prototype has three core features:
Image-to-text conversion: TextDrop uses a webcam to take cropped blackboard images. Each alphanumeric character on the blackboard is located, isolated, and formatted neatly on a PDF file.
Self-adjusting camera: TextDrop rotates the webcam at specific intervals to capture multiple blackboards in a lecture hall. It continuously follows the professor’s movements between boards to ensure all information is captured.
Document formatting: TextDrop transposes the captured characters onto a PDF file, where each blackboard in the lecture hall gets its own page. The document is neatly formatted to reflect the structure of the handwritten notes, including section headers, paragraph breaks, and bullet points.
For consistency, we captured images from the blackboards in the STC lecture halls.

## Project Implemented:

### What We Achieved:
Capturing photos from the Raspberry Pi
For photo capture, we used a Tomorsi 1080p HD Webcam. Every 5 seconds, our program sends an HTTP request to a server hosted on the Raspberry Pi, which tells it to capture an image and send it back to the user’s device.

### Motorized camera movement:
For comprehensive text capture across multiple chalkboards, we utilized an FS90R servo motor for automated camera rotation. The motor is programmed to rotate the camera to a new position, facing the next chalkboard, once the number of characters within an image has stabilized. This indicates that no new text has been added to the current chalkboard.

### Cropping and rectifying the image:
Since TextDrop was capturing images across multiple chalkboards, the photos were often taken at varying angles and zoom levels. These inconsistencies made it difficult to isolate text characters from the image and know which parts of the image to further process for character identification. Thus, we developed an image-processing algorithm that extracts the blackboard portion of any input photo. More specifically, it accurately identifies the blackboard region within a photo, crops out surrounding content, and normalizes and scales the image. This pre-processing step improved the quality of our subsequent character recognition algorithm.

<img width="276" alt="Screenshot 2025-01-06 at 6 54 30 PM" src="https://github.com/user-attachments/assets/0b3151b8-bd7f-44d9-a50d-b1c103982296" />

Figure 1: Original image without any modifications

<img width="285" alt="Screenshot 2025-01-06 at 6 54 43 PM" src="https://github.com/user-attachments/assets/5ed04dc0-476f-472a-a0c6-190b0c576852" />

Figure 2: Blackboard extraction during the image rectification process

<img width="284" alt="Screenshot 2025-01-06 at 6 55 01 PM" src="https://github.com/user-attachments/assets/c1be3f7c-72fe-4b32-9083-36f66b5d3ef2" />

Figure 3: Image after adjusting the angle and zoom level

### Character isolation:
To ensure the character recognition algorithm had consistent input, a program was written to isolate each character on the blackboard for evaluation. The script scans through the entire blackboard looking for potential characters. Once a character is found, it is bounded, isolated, and compressed into a smaller 28x28 pixel image to be passed on.

### Character recognition using machine learning:
We implemented a machine learning model to identify handwritten uppercase letters (A-Z) and digits (0-9) from the blackboard. First, the processed, isolated images of handwritten characters are passed into our model. Then, the model predicts characters with a 95% accuracy on validation data, which translates to an ~87% accuracy on STC blackboard test data.

<img width="234" alt="Screenshot 2025-01-06 at 6 55 15 PM" src="https://github.com/user-attachments/assets/31c9c994-2406-448e-826d-f4f58cd717d3" />

Figure 4: Prediction of a handwritten “D”

### Document formatting and PDF generation:
A multi-page PDF file is generated given the identified characters and their relative positions on the blackboard. Each blackboard gets its own page on the PDF. When the program detects very large text from a blackboard, it treats this as a section header on the PDF (larger, bold font).

<img width="329" alt="Screenshot 2025-01-06 at 6 55 39 PM" src="https://github.com/user-attachments/assets/3a8bc7c7-1a04-4eaa-b942-fc3d011077c3" />

Figure 5: Example of a generated PDF with a header and body text

### Custom 3D-printed case:
We designed a 3D-printed custom case made of PLA filament to securely house our hardware components (Raspberry Pi, camera, Servo motor) and organize our wiring. Additionally, it featured an adjustable stand to maintain a consistent and leveled perspective during image capture.

<img width="357" alt="Screenshot 2025-01-06 at 6 56 17 PM" src="https://github.com/user-attachments/assets/c02c65d1-6361-41a8-a0ff-68a10d207317" />

Figure 6: Model of 3D-printed case designed in Fusion

## Video:

#### Note:
The training dataset used for this project was the A-Z Kaggle dataset and 0-9 MNIST dataset from Keras.
