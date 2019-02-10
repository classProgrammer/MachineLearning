import numpy
import matplotlib.pyplot
import scipy.special
%matplotlib inline

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningRate):
        # initialize layers:
        self.inNodes = inputnodes #input layer
        self.hidNodes = hiddennodes #hidden layer
        self.outNodes = outputnodes #output layer
        
        # learning rate:
        self.lr = learningRate

        # connection state matrices:
        self.whi = numpy.random.normal(0.0, pow(self.hidNodes, -0.5), (self.hidNodes, self.inNodes))
        self.who = numpy.random.normal(0.0, pow(self.outNodes, -0.5), (self.outNodes, self.hidNodes))
        
        # sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
    def transpose(self, matrix):
        return numpy.array(matrix, ndmin=2).T
    
    def delta(self, learning_rate, output_error, output, previous_layer):
        return learning_rate * numpy.dot(
            (output_error * output * (1.0 - output)), 
            numpy.transpose(previous_layer)
        )
    
    def generateOutput(self, connection_state, inputs):
        return self.activation_function(
             numpy.dot(connection_state, inputs)
        )
        
    def train(self, inputs_list, targets_list):
        inputs = self.transpose(inputs_list)
        targets = self.transpose(targets_list)
        
        hidden_outputs = self.generateOutput(self.whi, inputs)
        final_outputs = self.generateOutput(self.who, hidden_outputs)
        
        # error backpropagation
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        self.who += self.delta(self.lr, output_errors, final_outputs, hidden_outputs)
        self.whi += self.delta(self.lr, hidden_errors, hidden_outputs, inputs)
       
    def query(self, inputs_list):
        # transposed input matrix
        inputs = self.transpose(inputs_list)
        
        # input layer -> hidden layer      
        hidden_outputs = self.generateOutput(self.whi, inputs)
        
        # hidden layer -> output layer
        final_outputs = self.generateOutput(self.who, hidden_outputs)
        
        return final_outputs

    