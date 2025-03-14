import numpy as np

def sigmoid(input):
    return 1 / (1 + np.exp(-input))


def list_to_vector(l):
        l1  = [[l[i]] for i in range(0, len(l))]
        m = np.matrix(l1)
        return m


def cost(result: list, desired_output: list):
        cost = 0
        for i in range(0, len(result)):
            cost += (result[i] - desired_output[0])**2
        return cost