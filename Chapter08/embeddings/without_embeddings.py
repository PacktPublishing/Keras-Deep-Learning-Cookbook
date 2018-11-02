from numpy import array
from keras.preprocessing.text import one_hot
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
labels = array([1,1,1,1,1,0,0,0,0,0])

vocab_size = 50
encodeDocuments = [one_hot(doc, vocab_size) for doc in documents]
print(encodeDocuments)

max_length = 4
paddedDocuments = pad_sequences(encodeDocuments, maxlen=max_length, padding='post')
print(paddedDocuments)

model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

model.fit(paddedDocuments, labels, epochs=50, verbose=0)

loss, accuracy = model.evaluate(paddedDocuments, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

