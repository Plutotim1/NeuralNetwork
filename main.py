import numpy as np
import random



class NeuralNetwork:
    def __init__(self, layer_count):
        if len(layer_count) <= 1:
            print("error: less than two layers")
            return;
        self.weights = {
            i:np.random.rand(
                layer_count[i],
                layer_count[i+1])
            for i in range(0, len(layer_count) - 1)}

        #for array in self.weights.values():
        #    print(array)




def main():
    nn = NeuralNetwork([10, 10, 10])
 


if __name__ == "__main__":
    main()