import numpy
from scipy.special import expit as sigmoid

class neuroLayer:
    def __init__(self, nodes, prevNodes):
        self.layer = numpy.random.normal(0.0, pow(nodes, -0.5), (nodes, prevNodes))
    
    def transposed(self):
        return self.layer.T
    
    def value(self):
        return self.layer

class neuroNet:
    def __init__(self, iNodes, hNodes, oNodes, learnRate, additional_layers = 0):
        self.lr = learnRate
        self.layers = []
        self.activation_function = lambda x: sigmoid(x)
        
        self.layers.append(neuroLayer(hNodes, iNodes))

        for _ in range(0, additional_layers):
            self.layers.append(neuroLayer(hNodes, hNodes))
        
        self.layers.append(neuroLayer(oNodes, hNodes))
        
    def updateNet(self, errors, outputs, inputs):
         return self.lr * numpy.dot((errors * outputs * (1.0 - outputs)), numpy.transpose(inputs))
        
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        Inputs = []
        Outputs = []
        Errors = []
        last = -1
        
        inp = inputs
        
        #inputs/outputs per layer
        for layer in self.layers:
            if len(Inputs) == 0:
                Inputs.append(inp) 
                inp = numpy.dot(layer.value(), inp)  
            else:
                Inputs.append(Outputs[last])
                inp = numpy.dot(layer.value(), Outputs[last])
                
            Outputs.append(self.activation_function(inp))
            last += 1
            
        #errors per layer (reverse order)
        lastLayer = None
        for layer in self.layers[::-1]:
            if len(Errors) == 0:
                Errors.append(targets - Outputs[last])
            else:
                Errors.insert(0, numpy.dot(lastLayer.layer.T, Errors[0]))
            lastLayer = layer

        idx = 0
        #update values per layer
        for layer in self.layers:
            layer.layer += self.updateNet(Errors[idx], Outputs[idx], Inputs[idx])
            idx += 1

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        for layer in self.layers:
            inputs = self.activation_function(numpy.dot(layer.value(), inputs))  
        return inputs