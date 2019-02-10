import numpy
import matplotlib.pyplot
import scipy.special
%matplotlib inline

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningRate):       
        # learning rate:
        self.lr = learningRate
        # connection state matrices:
        self.inputLayer = numpy.random.normal(0.0, pow(hiddennodes, -0.5), (hiddennodes, inputnodes))
		
		self.hiddenLayer = numpy.random.normal(0.0, pow(hiddennodes, -0.5), (hiddennodes, hiddennodes))
		
        self.outputLayer = numpy.random.normal(0.0, pow(outputnodes, -0.5), (outputnodes, hiddennodes))
        # sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
    def transpose(self, matrix):
        return numpy.array(matrix, ndmin=2).T
    
    def delta(self, learning_rate, output_error, output, previous_layer):
        return learning_rate * numpy.dot(
            (output_error * output * (1.0 - output)), 
            numpy.transpose(previous_layer)
        )
    
    def generateOutput(self, connectionStateMatrix, layerInput):
        return self.activation_function(
             numpy.dot(connectionStateMatrix, layerInput)
        )
        
    def train(self, input_list, target_list):
        input = self.transpose(input_list)
        target = self.transpose(target_list)
        
        hidden_output = self.generateOutput(self.inputLayer, input)
		hidden_output2 = self.generateOutput(self.hiddenLayer, hidden_output)
        final_output = self.generateOutput(self.outputLayer, hidden_output2)
        
        # error backpropagation
        output_error = target - final_output
        hidden_error2 = numpy.dot(self.outputLayer.T, output_error)
		hidden_error = numpy.dot(self.hiddenLayer.T, hidden_error2)
        
        self.outputLayer += self.delta(self.lr, output_error, final_output, hidden_output2)
		self.hiddenLayer += self.delta(self.lr, hidden_error2, hidden_output2, hidden_output)
        self.inputLayer += self.delta(self.lr, hidden_error, hidden_output, input)
       
    def query(self, input_list):
        # transposed input matrix
        input = self.transpose(input_list)
        
        # input layer -> hidden layer      
        hidden_output = self.generateOutput(self.inputLayer, input)
		
		# hidden layer 1 -> hidden layer 2    
        hidden_output2 = self.generateOutput(self.hiddenLayer, hidden_output)
        
        # hidden layer -> output layer
        final_output = self.generateOutput(self.outputLayer, hidden_output2)
        
        return final_output