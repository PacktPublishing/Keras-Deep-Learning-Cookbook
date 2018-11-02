from .helper import *
import numpy as np


class Word2Vec:
    def __init__(self, algo='skipgram', windowSize=2, numHidden=2, nEpochs=1, text='', learningRate=0.1):
        """
        initializer for word2vec
        :param type:
        :param windowSize:
        :param numHidden:
        :param nEpochs:
        :param text:
        :param learningRate:
        """
        self.algo = algo
        self.windowSize = windowSize
        self.numHidden = numHidden
        self.nEpochs = nEpochs
        self.text = text
        self.learningRate = learningRate

    def skipgram(self, context, center, weights1, weights2, loss):
        """
        skipgram model
        :param context:
        :param center:
        :param weights1:
        :param weights2:
        :param loss:
        :return:
        """
        h = np.dot(weights1.T, center)
        u = np.dot(weights2.T, h)
        yPred = softmax(u)

        error = np.array([yPred.T - label for label in context])
        dweights2 = np.outer(h, np.sum(error, axis=0))
        dweights1 = np.outer(center, np.dot(weights2, np.sum(error, axis=0)))

        newWeights1 = weights1 - self.learningRate * dweights1
        newWeights2 = weights2 - self.learningRate * dweights2

        loss += - np.sum([u[label == 1] for label in context]) + len(context) * np.log(np.sum(np.exp(u)))

        return newWeights1, newWeights2, loss

    def execute(self):
        """
        Executor for word2vec.
        :return:
        """
        if len(self.text) == 0:
            raise ValueError('text is expected')

        tokenizedText, size = tokenize(self.text)
        weights1, weights2 = setUp(size, self.numHidden)

        lossPerEpoch = []
        for epoch in range(self.nEpochs):
            loss = 0
            for context, center in getContexts(tokenizedText, size, self.windowSize):
                weights1, weights2, loss = self.skipgram(context, center, weights1, weights2, loss)
            lossPerEpoch.append(loss)

        return weights1, weights2, lossPerEpoch



