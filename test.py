import numpy
import NeuralNetwork
import data


def matrix_dimension():
    m1 = numpy.matrix([[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7]])
    v1 = numpy.matrix([[1],[1],[1],[1],[1]])
    result = numpy.matmul(m1, v1)
    print(result)


def weight_length():
    d = data.bigger_than_data(10)
    nn = NeuralNetwork.NeuralNetwork([5,2,89,2], data.Data(d[0],d[1]))


weight_length()