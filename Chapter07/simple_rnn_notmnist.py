'''This is a reproduction of the IRNN experiment
with pixel-by-pixel sequential MNIST in
"A Simple Way to Initialize Recurrent Networks of Rectified Linear Units"
by Quoc V. Le, Navdeep Jaitly, Geoffrey E. Hinton

arxiv:1504.00941v2 [cs.NE] 7 Apr 2015
http://arxiv.org/pdf/1504.00941v2.pdf

Optimizer is replaced with RMSprop which yields more stable and steady
improvement.

Reaches 0.93 train/test accuracy after 900 epochs
(which roughly corresponds to 1687500 steps in the original paper.)
'''

from __future__ import print_function
#import cPickle as pickle
import pickle
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras import initializers
from keras.optimizers import RMSprop

batch_size = 32
num_classes = 10
epochs = 20
hidden_units = 100

learning_rate = 1e-6
clip_norm = 1.0

# the data, split between train and test sets

from util import download_file

s3_url = 'https://s3.amazonaws.com/neural-networking-book/ch02/notMNIST_3.5.pickle?versionId=j53VUhZj_FXe9iFSN0O.KLedt08.DGy4'

pickle_file = download_file(s3_url, 'notMNSIT_3.5.pickle')
#pickle_file = './data/notMNIST.pickle'
image_size = 28
num_of_labels = 10
with open('./' + pickle_file, 'rb') as f:
    save = pickle.load(f)
    training_dataset = save['train_dataset']
    training_labels = save['train_labels']
    validation_dataset = save['valid_dataset']
    validation_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', training_dataset.shape, training_labels.shape)
    print('Validation set', validation_dataset.shape, validation_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_of_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels
#print(notmnist.data)

#train_dataset, train_labels = reformat(training_dataset, training_labels)
#valid_dataset, valid_labels = reformat(validation_dataset, validation_labels)
#test_dataset, test_labels = reformat(test_dataset, test_labels)
#print('Training dataset shape' + str(train_dataset.shape) +  "," + str(train_labels.shape))
#print('Validation dataset shape'+ str(valid_dataset.shape) + ',' +  str(valid_labels.shape))
#print('Test dataset shape' + str(test_dataset.shape) + ',' +  str(test_labels.shape))

x_train = training_dataset
y_train = training_labels
x_test = test_dataset
y_test = test_labels


x_train = x_train.reshape(x_train.shape[0], -1, 1)
x_test = x_test.reshape(x_test.shape[0], -1, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('Evaluate IRNN...')
model = Sequential()
model.add(SimpleRNN(hidden_units,
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    recurrent_initializer=initializers.Identity(gain=1.0),
                    activation='relu',
                    input_shape=x_train.shape[1:]))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

scores = model.evaluate(x_test, y_test, verbose=0)
print('IRNN test score:', scores[0])
print('IRNN test accuracy:', scores[1])

result = model.predict_classes(x_test[0], batch_size=1, verbose=0)
print(result)