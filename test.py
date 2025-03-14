import data


numbers = data.bigger_than_data(100, 7)
d = data.Data(numbers[0], numbers[1])
print(d.training_data)
print(d.get_batches(10))
