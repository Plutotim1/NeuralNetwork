import random
import help

class Data:
    def __init__(self, input, desired_output, test_ratio = 0.5):
        self.size = len(input)
        self.testing_data = [(input[i], desired_output[i]) for i in range(0, int(len(input) * test_ratio))]
        self.training_data = [(input[i], desired_output[i]) for i in range(int(len(input) * test_ratio), len(input))]
    

    def get_batches(self, batch_size):
        random.shuffle(self.training_data)
        l = len(self.training_data)
        if(l < batch_size):
            return -1
        result = []
        size = int(l / batch_size)
        for i in range(0, size):
            result.append(self.training_data[i * batch_size: (i+1) * batch_size])
        return result



def bigger_than_data(size):
    data = [[],[]]
    for i in range(0, size):
        n = random.random()
        m = random.random()
        data[0].append([n, m])
        data[1].append([1] if n > m else [0])
    return data


def seed_data():
    data = [[],[]]
    with open("seeds.txt") as file:
        for line in file:
            d = line.split()
            entry = []
            for i in range(7):
                entry.append(help.sigmoid(float(d[i])))
            
            output = [0, 0, 0]
            output[int(d[7]) - 1] = 1

            data[0].append(entry)
            data[1].append(output)            
    
    return data