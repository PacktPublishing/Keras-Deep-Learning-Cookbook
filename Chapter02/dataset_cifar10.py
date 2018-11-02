from keras.datasets import cifar10
from keras.datasets import cifar100


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("X_train shape: " + str(X_train.shape))
print("y_train shape: " + str(y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("y_test shape: " +  str(y_test.shape))