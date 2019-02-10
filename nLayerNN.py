import numpy
import matplotlib.pyplot
import scipy.special
%matplotlib inline

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningRate, hiddenLayers):	
        # learning rate:
        self.lr = learningRate

        # connection state matrices:
        self.whi = numpy.random.normal(0.0, pow(hiddennodes, -0.5), (hiddennodes, inputnodes))
        self.whh = numpy.random.normal(0.0, pow(hiddennodes, -0.5), (hiddennodes, hiddennodes))
        self.who = numpy.random.normal(0.0, pow(outputnodes, -0.5), (outputnodes, hiddennodes))
        
        # sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
    def transpose(self, matrix):
        return numpy.array(matrix, ndmin=2).T
    
    def delta(self, output_error, output, previous_layer):
        return self.lr * numpy.dot(
            (output_error * output * (1.0 - output)), 
            numpy.transpose(previous_layer)
        )
    
    def generateOutput(self, connectionStateMatrix, layerInputs):
        return self.activation_function(
             numpy.dot(connectionStateMatrix, layerInputs)
        )
        
    def train(self, inputs_list, targets_list):
        inputs = self.transpose(inputs_list)
        targets = self.transpose(targets_list)
        
        hidden_output_layer_1 = self.generateOutput(self.whi, inputs)
        hidden_output_layer_2 = self.generateOutput(self.whh, hidden_output_layer_1)
        final_outputs = self.generateOutput(self.who, hidden_output_layer_2)
        
        # error backpropagation
        output_errors = targets - final_outputs
        output_hidden_layer_2 = numpy.dot(self.who.T, output_errors)
        output_hidden_layer_1 = numpy.dot(self.whh.T, output_hidden_layer_2)
        
        self.who += self.delta(output_errors, final_outputs, output_hidden_layer_2)
        self.whh += self.delta(output_hidden_layer_2, hidden_output_layer_2, hidden_output_layer_1)
        self.whi += self.delta(output_hidden_layer_1, hidden_output_layer_1, inputs)
       
    def query(self, inputs_list):
        # transposed input matrix
        inputs = self.transpose(inputs_list)
        
        # input layer -> hidden layer      
        hidden_outputs = self.generateOutput(self.whi, inputs)
        
        # input layer -> hidden layer      
        hidden_outputs2 = self.generateOutput(self.whh, hidden_outputs)
        
        # hidden layer -> output layer
        final_outputs = self.generateOutput(self.who, hidden_outputs2)
        
        return final_outputs