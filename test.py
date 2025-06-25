import numpy
import NeuralNetwork
import data
import help


def matrix_dimension():
    m1 = numpy.matrix([[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7]])
    v1 = numpy.matrix([[1],[1],[1],[1],[1]])
    v2 = numpy.matrix([[1], [3], [5]])
    b1 = numpy.matrix([[2],[2],[2],[6],[90]])
    result = numpy.matmul(m1, v1)
    result2 = m1 + v2
    #print(result)
    #print(result2)
    print(1 - m1)


def sigmoid():
    m1 = numpy.matrix([[1,0,3,-1,-2], [2,3,4,5,6], [3,4,5,6,7]])
    print(help.sigmoid(m1))



def weights():
    d = data.bigger_than_data(100)
    nn = NeuralNetwork.NeuralNetwork([2,5,9,2], data.Data(d[0],d[1]))
    for i in range(3):
        print(i, ":")
        print(nn.biases[i])
        print(nn.weights[i])


def calculate():
    d = data.bigger_than_data(100)
    nn = NeuralNetwork.NeuralNetwork([2,1,1,1], data.Data(d[0],d[1]))
    for i in range(3):
        print(i, ":")
        print(nn.biases[i])
        print(nn.weights[i])
    print(nn.calculate([0.5, 0.7]))


def test_cost():
    d = data.bigger_than_data(100)
    nn = NeuralNetwork.NeuralNetwork([2,1,1,1], data.Data(d[0],d[1]))
    print(nn.test_cost())


def test_accuracy():
    d = data.bigger_than_data(100)
    nn = NeuralNetwork.NeuralNetwork([2,1,1,1], data.Data(d[0],d[1]))
    return nn.test_accuracy()


def test_average_accuracy():
    total = 0
    for i in range(100):
        total += test_accuracy()
    print(total / 100)


def train_batch():
    d = data.bigger_than_data(100)
    nn = NeuralNetwork.NeuralNetwork([2,1,1,1], data.Data(d[0],d[1]))
    batches = nn.data.get_batches(10)
    nn.train_batch(batches[0], 1)


def train():
    d = data.bigger_than_data(1000000)
    nn = NeuralNetwork.NeuralNetwork([2,1], data.Data(d[0],d[1], 0.1))
    nn.train(10, 1)


def train_visual():
    d = data.bigger_than_data(200)
    nn = NeuralNetwork.NeuralNetwork([2,1], data.Data(d[0],d[1], 0.1))
    nn.train_visual(10, 1)


#matrix_dimension()
#test_average_accuracy()
#train_batch()
#sigmoid()
#train()
train_visual()


