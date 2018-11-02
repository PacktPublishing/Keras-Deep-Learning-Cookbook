from numpy import array, asarray, zeros
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

# define documents
documents = ['Well done!',
             'Good work',
             'Great effort',
             'nice work',
             'Excellent!',
             'Weak',
             'Poor effort!',
             'not good',
             'poor work',
             'Could have done better.']

# define class labels
labels = array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# use tokenizer and pad
tokenizer = Tokenizer()
tokenizer.fit_on_texts(documents)
vocab_size = len(tokenizer.word_index) + 1
encodeDocuments = tokenizer.texts_to_sequences(documents)
print(encodeDocuments)

max_length = 4
paddedDocuments = pad_sequences(encodeDocuments, maxlen=max_length, padding='post')
print(paddedDocuments)

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

# create coefficient matrix for training data
trainingToEmbeddings = zeros((vocab_size, 100))
for word, i in tokenizer.word_index.items():
    gloveVector = inMemoryGlove.get(word)
    if gloveVector is not None:
        trainingToEmbeddings[i] = gloveVector

model = Sequential()
model.add(Embedding(vocab_size, 100, weights=[trainingToEmbeddings], input_length=max_length, trainable=False))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

model.fit(paddedDocuments, labels, epochs=50, verbose=0)

loss, accuracy = model.evaluate(paddedDocuments, labels, verbose=0)
print('Accuracy: %f' % (accuracy * 100))