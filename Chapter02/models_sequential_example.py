from keras.models import Sequential
from keras.layers import Dense, Activation


model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('tanh'),
    Dense(10),
    Activation('softmax'),
])

print(model.summary())