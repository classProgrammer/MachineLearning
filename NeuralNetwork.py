import numpy
import matplotlib.pyplot
import scipy.special
%matplotlib inline

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningRate):
        # initialize layers:
        self.inputLayer = inputnodes #input layer
        self.hiddenLayer1 = hiddennodes #hidden layer
		self.hiddenLayer2 = hiddennodes #hidden layer
        self.outputLayer = outputnodes #output layer
        
        # learning rate:
        self.lr = learningRate

        # connection state matrices:
        self.whi = numpy.random.normal(0.0, pow(self.hiddenLayer1, -0.5), (self.hiddenLayer1, self.inputLayer))
		self.whh = numpy.random.normal(0.0, pow(self.hiddenLayer2, -0.5), (self.hiddenLayer2, self.hiddenLayer1))
        self.who = numpy.random.normal(0.0, pow(self.outputLayer, -0.5), (self.outputLayer, self.hiddenLayer2))
        
        # sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
    def transpose(self, matrix):
        return numpy.array(matrix, ndmin=2).T
    
    def delta(self, output_error, output, previous_layer):
        return self.lr * numpy.dot(
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
		hidden_outputs2 = self.generateOutput(self.whh, hidden_outputs)
        final_outputs = self.generateOutput(self.who, hidden_outputs2)
        
        # error backpropagation
        output_errors = targets - final_outputs
        hidden_errors2 = numpy.dot(self.who.T, output_errors)
		hidden_errors1 = numpy.dot(self.whh.T, hidden_errors2)
		
        self.who += self.delta(output_errors, final_outputs, hidden_outputs2)
		self.whh += self.delta(hidden_errors2, hidden_outputs2, hidden_outputs)
        self.whi += self.delta(hidden_errors1, hidden_outputs, inputs)
       
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