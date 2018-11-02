from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
import numpy as np


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def setUp(length, numHidden):
    """
    Setup the models.
    :param length:
    :param size:
    :return:
    """
    np.random.seed(100)
    weights1 = np.random.rand(length, numHidden)
    weights2 = np.random.rand(numHidden, length)
    return weights1, weights2


def tokenize(text):
    """
    The Tokenizer stores everything in the word_index during fit_on_texts.
    Then, when calling the texts_to_sequences method, only the top num_words are considered.

    You can see that the value's are clearly not sorted after indexing.
    It is respected however in the texts_to_sequences method which turns input into numerical arrays:
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    tokenizedTextSequence = tokenizer.texts_to_sequences(text)
    size = len(tokenizer.word_index)
    return tokenizedTextSequence, size


def getContexts(tokenizedTextSequence, size, windowSize):
    """
    get context and center hot vectors
    :param tokenizedTextSequence:
    :param size:
    :param windowSize:
    :return:
    """
    for words in tokenizedTextSequence:
        length = len(words)
        for index, word in enumerate(words):
            contexts = []
            center = []
            start = index - windowSize
            end = index + windowSize + 1
            contexts.append([words[i]-1 for i in range(start, end) if 0 <= i < length and i != index])
            center.append(word - 1)
            x = to_categorical(contexts, size)
            y = to_categorical(center, size)
            yield (x, y.ravel())


def softmax(x):
    """
    Given an array of real numbers (including negative ones),
    the softmax function essentially returns a probability distribution with sum of the entries equal to one.
    :param x:
    :return:
    """
    e2x = np.exp(x - np.max(x))
    return e2x/e2x.sum(axis=0)


#def predictNextWord(word):
