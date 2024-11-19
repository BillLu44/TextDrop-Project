# This is the Convolutional Neural Network responsible for OCR.
# We are using the ResNet architecture.

from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K


class CNN:
    @staticmethod
    def residual_module(data, K, stride, channelDim, reduceSize=False, reg=0.0001, bnEps=2e-5, bnMom=0.9):
        shortcut = data

        # 1. Apply 1x1 filters (convolutions)
        bn1 = BatchNormalization(axis=channelDim, epsilon=bnEps, momentum=bnMom)(data)
        activated1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(activated1)

        # 2. Apply 3x3 filters (convolutions)
        bn2 = BatchNormalization(axis=channelDim, epsilon=bnEps, momentum=bnMom)(conv1)
        activated2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False, kernel_regularizer=l2(reg))(activated2)

        # 3. Apply another set of 1x1 filters
        bn3 = BatchNormalization(axis=channelDim, epsilon=bnEps, momentum=bnMom)(conv2)
        activated3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(activated3)

        # If size should be reduced, apply yet another 1x1 layer to the shortcut
        if reduceSize:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(activated1)

        # Add the final convolution and shortcut
        x = add([conv3, shortcut])
        return x
    
    @staticmethod
    def buildArch(width, height, depth, classes, stages, filters, reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset="cifar"):

        # Define the shape of the input tensor based on backend (K)
        inputShape = ()
        channelDim = 0

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            channelDim = 1
        else:    #channels_last
            inputShape = (height, width, depth)
            channelDim = -1

        inputs = Input(shape=inputShape)
        currTensor = BatchNormalization(axis=channelDim, epsilon=bnEps, momentum=bnMom)(inputs)

        # Initial convolution layer (based on the test dataset we are using)
        if dataset == "cifar":
            currTensor = Conv2D(filters[0], (3, 3), use_bias=False, padding="same", kernel_regularizer=l2(reg))(currTensor)

        elif dataset == "tiny_imagenet":
            # Convolution -> batch normalization -> activation -> max pooling (reduces size)
            currTensor = Conv2D(filters[0], (5, 5), use_bias=False, padding="same", kernel_regularizer=l2(reg))(currTensor)
            currTensor = BatchNormalization(axis=channelDim, epsilon=bnEps, momentum=bnMom)(currTensor)
            currTensor = Activation("relu")(currTensor)
            currTensor = ZeroPadding2D((1, 1))(currTensor)
            currTensor = MaxPooling2D((3, 3), strides=(2, 2))(currTensor)

        # Apply residual modules for each stage
        for i in range(0, len(stages)):
            stride = ()
            if i == 0:
                stride = (1, 1)
            else:
                stride = (2, 2)

            currTensor = CNN.residual_module(currTensor, filters[i + 1], stride, channelDim, reduceSize=True, bnEps=bnEps, bnMom=bnMom)

            # Apply a module for each layer in the stage
            for j in range(0, stages[i] - 1):
                currTensor = CNN.residual_module(currTensor, filters[i + 1], (1, 1), channelDim, bnEps=bnEps, bnMom=bnMom)

        
        # Batch normalization, activation, then average pooling
        currTensor = BatchNormalization(axis=channelDim, epsilon=bnEps, momentum=bnMom)(currTensor)
        currTensor = Activation("relu")(currTensor)
        currTensor = AveragePooling2D((8, 8))(currTensor)

        # Dense layer, then Softmax activation for classification
        currTensor = Flatten()(currTensor)
        currTensor = Dense(classes, kernel_regularizer=l2(reg))(currTensor)
        currTensor = Activation("softmax")(currTensor)

        # Return a Keras model made from the final tensor
        return Model(inputs, currTensor, name="resnet")
