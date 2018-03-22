# Based on MatEarl license plate char detector
# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import MaxPooling2D
from keras.layers import Reshape
from keras import backend as K
from keras.layers import Lambda
from keras.activations import softmax
from keras.layers import Dropout

class AnprCharDet:
  @staticmethod
  def build(height, width, depth, textLen, numCharClasses):
    # initialize the model along with the input shape to be
    # "channels last"
    model = Sequential()
    inputShape = (height, width, depth)

    # if we are using "channels first", update the input shape
    # eg input 64x128x1
    if K.image_data_format() == "channels_first":
      inputShape = (depth, height, width)

    # Conv Layer 1
    # define the first CONV => RELU => MAXPOOL layer
    # eg output 32x64x48
    model.add(Conv2D(48, (5, 5), padding="same",
                     input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Dropout
    model.add(Dropout(0.25))

    # Conv Layer 2
    # define the second layer CONV => RELU => MAXPOOL layer
    # eg output 16x64x64
    model.add(Conv2D(64, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    # Dropout
    model.add(Dropout(0.25))

    # Conv Layer 3
    # define the third layer CONV => RELU => MAXPOOL layer
    # eg output 8x32x128
    model.add(Conv2D(128, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Dropout
    model.add(Dropout(0.25))

    # Dense Layer 1
    # eg input 8x32x128 => 32768 => 2048
    model.add(Flatten())
    model.add(Dense(2048))
    # Dropout
    model.add(Dropout(0.5))

    # output Layer
    # eg (maxAlpha+maxDig) * maxLicPlateChars = (26+10)*7 = 252
    model.add(Dense(textLen * numCharClasses))
    # Dropout
    model.add(Dropout(0.5))
    model.add(Reshape((textLen, numCharClasses)))
    model.add(Activation('softmax'))
    #def mySoftmax(x):
    #  return softmax(x, axis=1)
    #model.add(Lambda(mySoftmax))
    
    # return the constructed network architecture
    return model

