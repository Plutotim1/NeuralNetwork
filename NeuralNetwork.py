import numpy
import help
import pygame
import sys


class NeuralNetwork:
    def __init__(self, layers : list[int], data):
        #initialize random weights
        self.layer_count = len(layers)
        self.weights = [
        
            numpy.matrix([
                [
                    help.signed_random() for j in range(layers[l])
                ] for i in range(layers[l + 1])
            ])
        for l in range(self.layer_count - 1)

        ]

        self.biases = [
        
            numpy.matrix([
                [
                    help.signed_random()
                ] for i in range(layers[l + 1])
            ])
         for l in range(self.layer_count - 1)
        ]

        self.layers = layers
        self.data = data

    
    def calculate(self, input : list[float]):
        #input first layer
        z_values = []

        input_vector = numpy.matrix([
                    [
                        input[i]
                    ] for i in range(self.layers[0])
                ])
        z_values.append(input_vector)

        #Feed forward
        for i in range(self.layer_count - 1):
            a_vector = numpy.matmul(self.weights[i], z_values[i]) + self.biases[i]
            z_values.append(
                help.sigmoid(a_vector)
            )
        
        return z_values[-1]
    

    def test_cost(self):
        total_cost = 0
        for dataset in self.data.testing_data:
            result = self.calculate(dataset[0])
            result_list = [result.item((i,0)) for i in range(result.shape[0])]
            total_cost += help.cost(result_list, dataset[1])
        average_cost = total_cost / len(self.data.testing_data)
        return average_cost


    def test_accuracy(self):
        false_count = 0
        total_count = 0
        for dataset in self.data.testing_data:
            total_count += 1
            result = self.calculate(dataset[0])
            result_list = [result.item((i,0)) for i in range(result.shape[0])]
            for i in range(len(result_list)):
                if (result_list[i] <= 0.5 and dataset[1][i] == 1) or (result_list[i] >= 0.5 and dataset[1][i] == 0):
                    false_count += 1
                    break
        ratio = (total_count - false_count) / total_count
        return ratio
    

    def train_batch(self, batch, learning_rate):
        batch_size = len(batch)
        
        desired_output = numpy.matrix(
            [
                [
                    batch[j][1][i] for j in range(batch_size)
                ] for i in range(self.layers[-1])
            ]
        )

        #Feed Forward
        z_matrizes = []
        input_matrix = numpy.matrix(
            [
                [
                    batch[j][0][i] for j in range(batch_size)
                ] for i in range(self.layers[0])
            ]
        )

        z_matrizes.append(input_matrix)

        for i in range(self.layer_count - 1):
            a_matrix = numpy.matmul(self.weights[i], z_matrizes[i]) + self.biases[i]
            z_matrizes.append(
                help.sigmoid(a_matrix)
            )
        
        #Backpropagation-Error

        error_matrizes = []

        #Last Error
        sigmoid_derivative = numpy.multiply(z_matrizes[-1], 1 - z_matrizes[-1])
        last_error = numpy.multiply((z_matrizes[-1] - desired_output), sigmoid_derivative)
        error_matrizes.insert(0, last_error)

        #Remaining Errors
        for i in range(self.layer_count - 2, -1, -1):
            sigmoid_derivative = numpy.multiply(z_matrizes[i], 1 - z_matrizes[i])
            error_matrix = numpy.multiply(numpy.matmul(self.weights[i].T, error_matrizes[0]), sigmoid_derivative)
            error_matrizes.insert(0, error_matrix)

        #adapting biases and weights
        for i in range(self.layer_count - 1):
            #biases
            vector1 = numpy.matrix([
                [1] for i in range(batch_size)
            ])
            average_bias_gradient = numpy.matmul(error_matrizes[i+1], vector1)
            self.biases[i] -= average_bias_gradient * (learning_rate / batch_size)

            #weights
            average_weight_gradient = numpy.matmul(error_matrizes[i + 1], z_matrizes[i].T)
            #print(average_weight_gradient)
            #print(average_bias_gradient)
            self.weights[i] -= average_weight_gradient * (learning_rate / batch_size)



        


    def train(self, batch_size = 10, learning_rate = 1, epochs = 1):
        batches = self.data.get_batches(batch_size)
        print("initial accuraccy:")
        print(self.test_accuracy())
        print("cost:")
        print(self.test_cost())
        for e in range(epochs):
            for batch in batches:
                self.train_batch(batch, learning_rate)
                #print("new_accuracy:")
                #print(self.test_cost())
                #print(self.test_accuracy())
        print("final accuracy:")
        print(self.test_accuracy())
        print("cost:")
        print(self.test_cost())


    #only for a 2-1 Network
    def train_visual(self, batch_size = 10, learning_rate = 1):

        #graphics
        pygame.init()
        size = (1000,600)
        white = 255, 255, 255
        gray = 100, 100, 100
        iterations = self.data.size / batch_size
        duration = 10000
        quit = False
        


        batches = self.data.get_batches(batch_size)
        print("initial accuraccy:")
        print(self.test_accuracy())

        screen = pygame.display.set_mode(size)
        next = False
        for batch in batches:
            screen.fill(white)
            
            self.train_batch(batch, learning_rate)

            #draw nodes
            pygame.draw.circle(screen, gray, (300, 150), 50)
            pygame.draw.circle(screen, gray, (300, 450), 50)
            pygame.draw.circle(screen, gray, (700, 300), 50)

            w = self.weights[0]
            b = self.biases[0].flat[0]

            pygame.draw.line(screen, help.value_to_color(w.flat[0]), (300, 150), (700, 300), 10)
            pygame.draw.line(screen, help.value_to_color(w.flat[1]), (300, 450), (700, 300), 10)
            pygame.draw.circle(screen, help.value_to_color(b), (700, 300), 30)

            ac = self.test_accuracy()
            
            pygame.draw.line(screen, (0, 0, 0), (100, 50), (900, 50), 5)
            pygame.draw.rect(screen, (0, 0, 0), (100 + int(ac * 800), 45, 10, 10))

            pygame.display.flip()
            pygame.time.delay(int(duration / iterations))

            while True:
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: sys.exit()
                    elif event.type == pygame.MOUSEBUTTONDOWN: next = True
                if (next): break
            

        
        print("training finished")

        
        print("final accuracy:")
        print(self.test_accuracy())


        while True:
                next = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: sys.exit()


    #for any dimension
    def train_any_visual(self, batch_size = 10, learning_rate = 1, epochs = 1, duration = 5000):

        #graphics
        pygame.init()
        size = (2000,1200)
        white = 255, 255, 255
        gray = 100, 100, 100
        iterations = self.data.size * epochs / (batch_size)
        quit = False
        


        batches = self.data.get_batches(batch_size)
        print("initial accuraccy:")
        print(self.test_accuracy())
        print("cost:")
        print(self.test_cost())

        screen = pygame.display.set_mode(size)
        next = False
        for e in range(epochs):
            #print("epoch:", e)
            for batch in batches:
                screen.fill(white)
            
                self.train_batch(batch, learning_rate)

                #draw nodes and biases
                x_dist = size[0] / (self.layer_count + 1)
                for i in range(self.layer_count):
                    y_dist = (size[1] - 200) / (self.layers[i] + 1)
                    if (i != 0):
                        b = self.biases[i - 1].flat
                    for j in range(self.layers[i]):
                        pygame.draw.circle(screen, gray, (x_dist * (i + 1), ((j + 1) * y_dist) + 200), 40)
                        if (i != 0):
                            pygame.draw.circle(screen, help.value_to_color(b[j]), (x_dist * (i + 1), ((j + 1) * y_dist) + 200), 30)

                    if (i == 0):
                        continue
                    #draw weights
                    w = self.weights[i - 1]
                    for start in range(self.layers[i - 1]):
                        for end in range(self.layers[i]):
                            start_p = (x_dist * (i), ((start + 1) * ((size[1] - 200) / (self.layers[i - 1] + 1))) + 200)
                            end_p = (x_dist * (i + 1), ((end + 1) * y_dist) + 200)
                            #print(start, end)
                            #print(type(w[start,end]))
                            #print(w[start,end])
                            pygame.draw.line(screen, help.value_to_color(w[end,start]), start_p, end_p, 6)

                #show accuracy
                ac = self.test_accuracy()
            
                pygame.draw.line(screen, (0, 0, 0), (200, 100), (size[0] - 200, 100), 5)
                pygame.draw.rect(screen, (255, 0, 0), (200 + int(ac * (size[0] - 400)), 95, 10, 10))

                #show cost
                c = self.test_cost()
            
                pygame.draw.line(screen, (0, 0, 0), (200, 150), (size[0] - 200, 150), 5)
                pygame.draw.rect(screen, (255, 0, 0), (200 + int(c * (size[0] - 400)), 145, 10, 10))

                pygame.display.flip()
                pygame.time.delay(int(duration / iterations))

                while True:
                
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT: sys.exit()
                        elif event.type == pygame.MOUSEBUTTONDOWN: next = True
                    if (next): break
            

        
        print("training finished")

        
        print("final accuracy:")
        print(self.test_accuracy())
        print("cost:")
        print(self.test_cost())


        while True:
                next = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: sys.exit()
        



