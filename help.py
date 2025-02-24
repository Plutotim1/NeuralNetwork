import numpy as np

def sigmoid(input):
    return 1 / (1 + np.exp(-input))

def list_to_vector(l):
        l1  = [[l[i]] for i in range(0, len(l))]
        m = np.matrix(l1)
        return m