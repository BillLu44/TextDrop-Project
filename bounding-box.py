import cv2
import numpy as np
from matplotlib import pyplot as plt

import sys
sys.setrecursionlimit(10000)

blackBoard = cv2.imread("TestImage3.JPG", cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(blackBoard, cv2.COLOR_BGR2GRAY)

width = np.size(gray_image, 1)
height = np.size(gray_image, 0)

def dfs(x, y, connectedCmp, depth):
    box = [x, y, x, y]
    if(depth > 9000):
        return box
    for i in range(-10, 11):
        for j in range (-10, 11):
            newX = x + i
            newY = y + j
            if(0 <= newX < height and 0 <= newY < width):
                if(gray_image[newX, newY] > threshold and vis[newX, newY] == 0):
                    vis[newX, newY] = connectedCmp
                    newBox = dfs(newX, newY, connectedCmp, depth+1)
                    box[0] = max(box[0], newBox[0])
                    box[1] = max(box[1], newBox[1])
                    box[2] = min(box[2], newBox[2])
                    box[3] = min(box[3], newBox[3])
    return box

def makeNNdata(box, connectedComp):
    bounded = np.copy(gray_image[box[2]-1:box[0]+1, box[3]-1:box[1]+1])
    boundedW = np.size(bounded, 1)
    boundedH = np.size(bounded, 0)
    for x in range(boundedH):
        for y in range(boundedW):
            if(vis[x + box[2] - 1, y + box[3] - 1] == connectedCmp):
                bounded[x, y] = 255
            else:
                bounded[x, y] = 0   
    s = max(bounded.shape)
    squared = np.zeros((s,s), dtype = 'float32')
    squared[(s - boundedH) // 2: (s - boundedH) // 2 + boundedH, (s - boundedW) // 2: (s - boundedW) // 2 + boundedW] = bounded
    nnBox = cv2.resize(squared, (28, 28), cv2.INTER_AREA)
    plt.imshow(nnBox)
    plt.title("box")
    plt.show()

def makeBoxes():
    connectedComp = 2
    vis = np.zeros((height, width), dtype = 'int32')
    threshold = 220

    for x in range(height):
        for y in range(width):
            if(vis[x, y] == 0):
                if(gray_image[x, y] > threshold):
                    vis[x, y] = connectedComp
                    box = dfs(x, y, connectedComp, 0)
                    # if(box[0] - box[2] >= 15 and box[1] - box[3] >= 15):
                    #     showBox(box, connectedComp)
                    connectedComp = connectedComp + 1
                else:
                    vis[x, y] = 1