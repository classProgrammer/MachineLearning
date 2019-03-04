import numpy
from FileRead import readFromFile
from NeuroNet import neuroNet

def trainNumberNet(net, data):
    for record in data:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        net.train(inputs, targets)

input_nodes = 784
hidden_nodes = 80
output_nodes = 10
learning_rate = 0.25
additional_layers = 1

n = neuroNet(input_nodes, hidden_nodes, output_nodes, learning_rate, additional_layers)

training_data_list = readFromFile("../dat/mnist_train.csv")

trainNumberNet(n, training_data_list)
print("trained")