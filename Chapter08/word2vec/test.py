from ch08.word2vec.word2vec import Word2Vec
import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np

corpus = ["I like playing football with my friends"]
skipgram = Word2Vec(algo="skipgram", text=corpus, windowSize=1, numHidden=2, nEpochs=600, learningRate=0.1)
weights1, weights1, lossPerEpoch = skipgram.execute()

print(len(lossPerEpoch))

plt.plot(np.arange(0, 600, 1), lossPerEpoch)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Skipgram')
plt.savefig("test.png")
plt.show()


