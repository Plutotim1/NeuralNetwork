import numpy
import NeuralNetwork
import data


def matrix_dimension():
    m1 = numpy.matrix([[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7]])
    v1 = numpy.matrix([[1],[1],[1],[1],[1]])
    result = numpy.matmul(m1, v1)
    print(result)


def weights():
    d = data.bigger_than_data(10)
    nn = NeuralNetwork.NeuralNetwork([2,5,9,2], data.Data(d[0],d[1]))
    for i in range(3):
        print(i, ":")
        print(nn.biases[i])
        print(nn.weights[i])


weights()


