import numpy as np
import help


class NeuralNetwork:
    def __init__(self, layer_count: list):
        self.layer_count = layer_count
        if len(layer_count) <= 1:
            print("error: less than two layers")
            return

        self.weights = [
            np.matrix([
                [(np.random.rand() - 0.5) * 2 for k in range(0, layer_count[i])]
                for j in range(0, layer_count[i + 1])
            ])
            for i in range(0, len(layer_count) - 1)
        ]

        '''
        self.weights = [
            np.random.rand(
                layer_count[i + 1],
                layer_count[i])
            for i in range(0, len(layer_count) - 1)]
        '''

        self.biases = [
            np.matrix([
            [(np.random.rand() - 0.5) * 2]
            for j in range(0, layer_count[i])])
            for i in range(1, len(layer_count))
        ]
        
        '''
        self.biases = [
            np.random.rand(
                layer_count[i],
                1
            ) for i in range(1, len(layer_count))
        ]
        '''
        


    def calculate_result(self, input: list):
        if len(input) != self.layer_count[0]:
            return [-1 for i in range(0, self.layer_count[-1])]

        prev = help.list_to_vector(input);

        for i in range(0, len(self.layer_count) - 1):
            

            temp = (self.weights[i] * prev) + self.biases[i]
            #print(temp)
            prev = help.sigmoid(temp)
            #print(prev)
            #print("SEPERATOR_________________________________________________________________________________")

        return prev;
            
        
    def print(self):
        for i in range(0, len(self.layer_count) - 1):
            print(self.biases[i])
            print(self.weights[i])