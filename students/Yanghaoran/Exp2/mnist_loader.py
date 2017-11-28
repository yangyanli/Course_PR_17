

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

def ReadData(filename):
    file = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(file,encoding="latin-1")
    file.close()
    return training_data, validation_data, test_data

def ChangeData(filename):

    training_data, validation_data, test_data = ReadData(filename)

    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    print(training_inputs[0])
    training_outputs = [change(y) for y in training_data[1]]

    training_data = list(zip(training_inputs, training_outputs))
    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]

    validation_data = list(zip(validation_inputs, validation_data[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))

    return training_data, validation_data, test_data

def change(j):

    result = np.zeros((10, 1))
    result[j] = 1.0
    return result

