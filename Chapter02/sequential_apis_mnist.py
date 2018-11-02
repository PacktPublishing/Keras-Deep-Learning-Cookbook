from keras.utils import plot_model
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.datasets import mnist
import keras

num_classes = 10
batch_size = 32
epochs = 10
batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# input layer
model = Sequential([
  Flatten(input_shape=(28, 28)),
  Dense(32, input_dim=784),
  Activation("sigmoid"),
  Dense(10),
  Activation("softmax"),
])

# summarize layers
print(model.summary())

# plot graph
plot_model(model, to_file='shared_input_layer.png')

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])