import numpy
import help


class NeuralNetwork:
    def __init__(self, layers : list[int], data):
        #initialize random weights
        self.weights = [
            [
                numpy.matrix([
                    [
                        help.signed_random() for j in range(layers[l])
                    ] for i in range(layers[l + 1])
                ])
            ] for l in range(len(layers) - 1)

        ]
        
        self.biases = [
            [
                numpy.matrix([
                    [
                        help.signed_random()
                    ] for i in range(layers[l + 1])
                ])
            ] for l in range(len(layers) - 1)
        ]


        self.data = data


