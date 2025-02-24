import numpy as np
import help


class NeuralNetwork:
    def __init__(self, layer_count):
        self.layer_count = layer_count
        if len(layer_count) <= 1:
            print("error: less than two layers")
            return;

        self.weights = [
            np.random.rand(
                layer_count[i + 1],
                layer_count[i])
            for i in range(0, len(layer_count) - 1)]
        
        self.biases = [
            np.random.rand(
                layer_count[i],
                1
            ) for i in range(1, len(layer_count))
        ]

    def list_to_vector(self, l):
        l1  = [[l[i]] for i in range(0, len(l))]
        m = np.matrix(l1)
        return m


    def calculate_result(self, input):
        if len(input) != len(self.weights[0][0]):
            return [-1 for i in range(0, self.layer_count[-1])]

        prev = self.list_to_vector(input);

        for i in range(0, len(self.layer_count) - 1):
            

            temp = (self.weights[i] * prev) + self.biases[i]
            print(temp)
            prev = help.sigmoid(temp)
            print(prev)
            print("SEPERATOR_________________________________________________________________________________")

        return prev;
            
        
    def print(self):
        for i in range(0, len(self.layer_count) - 1):
            print(self.biases[i])
            print(self.weights[i])
        
        





def main():
    nn = NeuralNetwork([2, 3, 2])
    print(nn.calculate_result([0.5, 1]))
    #nn.print()

 


if __name__ == "__main__":
    main()