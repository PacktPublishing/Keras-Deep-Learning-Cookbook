from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

# get dataset
data = pd.read_csv('./spam_dataset.csv')
texts = []
classes = []
for i, label in enumerate(data['Class']):
    texts.append(data['Text'][i])
    if label == 'ham':
        classes.append(0)
    else:
        classes.append(1)

texts = np.asarray(texts)
classes = np.asarray(classes)

print("number of texts :", len(texts))
print("number of labels: ", len(classes))

# number of words used as features
maxFeatures = 10000
# max document length
maxLen = 500

# we will use 80% of data as training and 20% as validation data
trainingData = int(len(texts) * .8)
validationData = int(len(texts) - trainingData)

# tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print("Found {0} unique words: ".format(len(word_index)))
data = pad_sequences(sequences, maxlen=maxLen)
print("data shape: ", data.shape)

np.random.seed(42)
# shuffle data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = classes[indices]

X_train = data[:trainingData]
y_train = labels[:trainingData]
X_test = data[trainingData:]
y_test = labels[trainingData:]


# modeling
model = Sequential()
model.add(Embedding(maxFeatures, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
rnn = model.fit(X_train, y_train, epochs=10, batch_size=60, validation_split=0.2)

# predictions
pred = model.predict_classes(X_test)
acc = model.evaluate(X_test, y_test)
proba_rnn = model.predict_proba(X_test)
print("Test loss is {0:.2f} accuracy is {1:.2f}  ".format(acc[0],acc[1]))
print(confusion_matrix(pred, y_test))

