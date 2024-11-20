import cv2
import numpy as np


def image_processor(x):

    # Reads an image stored on a computer.
    image = cv2.imread(x)

    # Converts the image into a grayscale image.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applies a Gaussian blur to reduce background noise.
    blurred = cv2.GaussianBlur(gray, (53, 53), 0)

    # Converts the image to black and white.
    retval, binary_image = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)

    # Use Canny edge detection to find edges
    edges = cv2.Canny(binary_image, 80, 150)

    # Find contours in the edge image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Takes the largest contour as the blackboard.
    largest_contour = max(contours, key=cv2.contourArea)

    # Error check.
    if not contours:
        print("No contours found.")
        exit()

    # Create a mask to extract the blackboard area.
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask)

    # Get the bounding box of the largest contour.
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image to the bounding box.
    cropped_image = masked_image[y:y + h, x:x + w]

    return cropped_image
