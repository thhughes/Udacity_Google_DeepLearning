import numpy as np

def softmax(x):
    """ Takes a List (or np.List) and returns a np.List of the softmax output"""
    npX = np.array(x)
    expX = np.exp(x)

    return expX/sum(expX)

