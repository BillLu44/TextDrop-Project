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
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=12(reg))(activated1)

        # 2. Apply 3x3 filters (convolutions)
        bn2 = BatchNormalization(axis=channelDim, epsilon=bnEps, momentum=bnMom)(conv1)
        activated2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False, kernel_regularizer=12(reg))(activated2)

        # 3. Apply another set of 1x1 filters
        bn3 = BatchNormalization(axis=channelDim, epsilon=bnEps, momentum=bnMom)(conv2)
        activated3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=12(reg))(activated3)

        # If size should be reduced, apply yet another 1x1 layer to the shortcut
        if reduceSize:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=12(reg))(activated1)

        # Add the final convolution and shortcut
        x = add([conv3, shortcut])
        return x
    
    