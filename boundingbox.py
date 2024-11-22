import cv2
import numpy as np

# dfs for detecting maximal connected component
def dfs(x, y, connectedCmp, width, height, gray_image, vis, threshold):

    # declare bounding box and size of contained blob
    box = [x, y, x, y, 1]

    # loop through nearby pixels
    for i in range(-10, 11):
        for j in range (-10, 11):
            newX = x + i
            newY = y + j

            # if adjacent pixel is in image and passes threshold and has not been previously visited, call dfs again
            if(0 <= newX < height and 0 <= newY < width):
                if(gray_image[newX, newY] > threshold and vis[newX, newY] == 0):

                    # label which connected component
                    vis[newX, newY] = connectedCmp

                    # call dfs recursively
                    newBox = dfs(newX, newY, connectedCmp, width, height, gray_image, vis, threshold)
                    # transfer bounding box data
                    box[0] = max(box[0], newBox[0])
                    box[1] = max(box[1], newBox[1])
                    box[2] = min(box[2], newBox[2])
                    box[3] = min(box[3], newBox[3])
                    box[4] = box[4] + newBox[4]

                    # check to minimize runtime (letters are typically <3000 pixels)
                    if(box[4] > 5000):
                        return box
    return box

# takes a bounding box and returns a data point of neural network
def makeNNdata(box, connectedCmp, gray_image, vis):

    # make sub image
    bounded = np.copy(gray_image[box[2]-1:box[0]+1, box[3]-1:box[1]+1])
    boundedW = np.size(bounded, 1)
    boundedH = np.size(bounded, 0)

    # black out anything in bounding box that isn't the corredsponding connected component
    # whiten everything in correct component to maximize contrast
    for x in range(boundedH):
        for y in range(boundedW):
            if(vis[x + box[2] - 1, y + box[3] - 1] == connectedCmp):
                bounded[x, y] = 255
            else:
                bounded[x, y] = 0

    # Reshape sub_image to 28 by 28 pixels
    s = max(bounded.shape)
    squared = np.zeros((s,s), dtype = 'float32')
    squared[(s - boundedH) // 2: (s - boundedH) // 2 + boundedH, (s - boundedW) // 2: (s - boundedW) // 2 + boundedW] = bounded
    nnBox = cv2.resize(squared, (28, 28), cv2.INTER_AREA)
    nnBox = np.reshape(nnBox, (1, 28, 28))
    return nnBox

    # debugging code
    # print(box[4])
    # plt.imshow(nnBox)
    # plt.title("box")
    # plt.show()

def load_data(gray_image):
    width = np.size(gray_image, 1)
    height = np.size(gray_image, 0)

    # declare threshold and visited array
    connectedComp = 2
    # Codes: 0 means not visited, 1 means does not exceed threshold, >= 2 means visited and vis[x][y] denotes which connected component (x, y) is in
    vis = np.zeros((height, width), dtype = 'int32')
    threshold = 220
    data = np.zeros((0, 28, 28), dtype = 'float32')
    boxes = np.zeros((0, 5), dtype = 'int32')

    # Loop through image by pixel (every ten pixels and hope you hit all letters)
    for x in range(0, height, 10):
        for y in range(0, width, 10):
            if(vis[x, y] == 0):
                if(gray_image[x, y] > threshold):
                    vis[x, y] = connectedComp
                    box = dfs(x, y, connectedComp, width, height, gray_image, vis, threshold)

                    # Letters are typically 600 to 3000 pixels, this minimized garbage components in data
                    if(box[4] >= 200 and box[4] <= 5000):
                        data = np.append(data, makeNNdata(box, connectedComp, gray_image, vis), axis = 0)
                        boxes = np.append(boxes, np.reshape(box, (1,5)), axis = 0)

                    connectedComp = connectedComp + 1
                else:
                    vis[x, y] = 1
    return (boxes, data)