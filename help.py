import numpy as np

def sigmoid(input):
    return 1 / (1 + np.exp(-input))