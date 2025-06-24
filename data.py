import random

class Data:
    def __init__(self, input, desired_output):
        self.testing_data = [(input[i], desired_output[i]) for i in range(0, int(len(input) / 2))]
        self.training_data = [(input[i], desired_output[i]) for i in range(int(len(input) / 2), len(input))]
    

    def get_batches(self, batch_size):
        random.shuffle(self.testing_data)
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