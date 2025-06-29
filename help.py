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
            cost += 0.5 * (result[i] - desired_output[i])**2
        return cost

def signed_random():
      return ((np.random.rand() - 0.5) * 2) / 1


def value_to_color(val):
      int_value = int(sigmoid(val) * 255)
      return (int_value, 0, 255 - int_value)

def accuracy_to_color(val):
      int_value = int(sigmoid(val) * 255)
      print (int_value, 0, 255 - int_value)
      return (255 - int_value, int_value, 0)
