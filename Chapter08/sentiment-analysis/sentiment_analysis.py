import csv
from numpy import array, asarray, zeros
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, SpatialDropout1D, LSTM
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
import pandas as pd
import re
from sklearn.model_selection import train_test_split

# read input document
X = pd.read_csv('/Users/manpreet.singh/git/deeplearning/deeplearning-keras/ch08/sentiment-analysis/Sentiment.csv')
X = X[['text', 'sentiment']]
X = X[X.sentiment != 'Neutral']
X['text'] = X['text'].apply(lambda x: x.lower())
X['text'] = X['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
print(X)

for idx, row in X.iterrows():
    row[0] = row[0].replace('rt', ' ')

print(type(X))

# use tokenizer and pad
maxFeatures = 2000
tokenizer = Tokenizer(num_words=maxFeatures, split=' ')
tokenizer.fit_on_texts(X['text'].values)
encodeDocuments = tokenizer.texts_to_sequences(X['text'].values)
print(encodeDocuments)

max_length = 29
paddedDocuments = pad_sequences(encodeDocuments, maxlen=max_length, padding='post')

# load glove model
inMemoryGlove = dict()
f = open('/Users/manpreet.singh/git/deeplearning/deeplearning-keras/ch08/embeddings/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefficients = asarray(values[1:], dtype='float32')
    inMemoryGlove[word] = coefficients
f.close()
print(len(inMemoryGlove))

# split data
labels = []
for i in X['sentiment']:
    if i == 'Positive':
        labels.append(1)
    else:
        labels.append(0)

labels = array(labels)

X_train, X_test, Y_train, Y_test = train_test_split(paddedDocuments,labels, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# create coefficient matrix for training data
trainingToEmbeddings = zeros((maxFeatures, 100))
for word, i in tokenizer.word_index.items():
    if i < 2001:
        gloveVector = inMemoryGlove.get(word)
        if gloveVector is not None:
            trainingToEmbeddings[i] = gloveVector

model = Sequential()
model.add(Embedding(maxFeatures, 100, weights=[trainingToEmbeddings], input_length=max_length, trainable=False))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

batch_size = 32
model.fit(X_train, Y_train, epochs=50, batch_size=batch_size, verbose=0)

loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print('Accuracy: %f' % (accuracy * 100))