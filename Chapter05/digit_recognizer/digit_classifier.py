import numpy
from keras import backend as K
from keras.utils import np_utils
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers.core import Activation

K.set_image_data_format('channels_last')
numpy.random.seed(0)

# get dataset
(XTrain, yTrain), (XTest, yTest) = mnist.load_data()
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(XTrain[1], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(XTrain[2], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(XTrain[3], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(XTrain[4], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

# create sequential model
model = Sequential()

# normalize the dataset
XTrain = XTrain / 255
XTest = XTest / 255

# data exploration
print("Number of training examples = %i" % XTrain.shape[0])
print("Number of classes = %i" % len(numpy.unique(yTrain)))
print("Dimension of images = {:d} x {:d}  ".format(XTrain[1].shape[0], XTrain[1].shape[1]))
unique, count = numpy.unique(yTrain, return_counts=True)
print("The number of occurrences of each class in the dataset = %s " % dict(zip(unique, count)), "\n")

XTrain = XTrain.reshape(XTrain.shape[0], 28, 28, 1).astype('float32')
XTest = XTest.reshape(XTest.shape[0], 28, 28, 1).astype('float32')
yTrain = np_utils.to_categorical(yTrain)
yTest = np_utils.to_categorical(yTest)

# modeling
model.add(Conv2D(40, kernel_size=5, padding="same", input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(50, kernel_size=5, padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(XTrain, yTrain, epochs=32, batch_size=200, validation_split=0.2)
scores = model.evaluate(XTest, yTest, verbose=10)
print(scores)
