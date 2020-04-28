from random import seed
from random import random
from math import exp
import numpy as np




def initialize_network(n_inputs,n_hidden,n_outputs):
    network=list()
    hidden_layer=[{'weights':[0 for i in range(n_inputs+1)]}for i in range(n_hidden)]
    network.append(hidden_layer)
    hidden_layer=[{'weights':[0 for i in range(n_inputs+1)]}for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer=[{'weights':[0 for i in range(n_hidden+1)]}for i in range(n_outputs)]
    network.append(output_layer)
    return network


def activate(weights,inputs):
    activation=weights[-1]
    for i in range(len(weights)-1):
        activation+=weights[i]*inputs[i]
    return activation

def transfer(activation):
    #print (1.0/(1.0+exp(-activation)))
    return 1.0/(1.0+np.exp(-activation))

def transfer_derivative(output):
	return output * (1.0 - output)

def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])



def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

def update_weights(network, row, l_rate,d,n_epoch):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += (l_rate/1+(l_rate/d)*n_epoch) * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += (l_rate/1+(l_rate/d)*n_epoch) * neuron['delta']

def train_network(network, train, l_rate, n_epoch, n_outputs,d):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate,d,n_epoch)
		#print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

def predict(network,row):
    outputs=forward_propagate(network,row)
    return outputs.index(max(outputs))

def main():
    training=np.loadtxt('train.csv',delimiter=',')
    training=training.astype(int)
    x=training[:,0:4]
    y=training[:,4]
    testing=np.loadtxt('test.csv',delimiter=',')
    testing=testing.astype(int)
    testx=testing[:,0:4]
    testy=testing[:,4]
    inputs=len(training[0]-1)
    outputs=len(set(row[-1]for row in training))
    seed(1)
    network=initialize_network(inputs,5,outputs)
    train_network(network,training,0.08,500,outputs,1000)
    s=0
    for row in training:
        prediction=predict(network,row)
        #print('Expected=%d, Got=%d' % (row[-1], prediction))
        if(row[-1]==prediction):
            s+=1
    print(s)
    print("Training Accuracy is: "+str(100*(s/872.0)))
    s=0
    for row in testing:
        prediction=predict(network,row)
        #print('Expected=%d, Got=%d' % (row[-1], prediction))
        if(row[-1]==prediction):
            s+=1
    print(s)
    print("Testing Accuracy is: "+str(100*(s/500.0)))






if __name__=="__main__":
    main()