#Creating a neural network class
import numpy as np
import math

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
#a bit unclear about this
def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        #init is the constructor, to create objects
        self.input  = x
        #the value of 'input' in the object that self refers to is x (the input is x)
        self.weights1  = np.random.rand(self.input.shape[1],4)
        #the value of 'weights1' is an array of one dimension with 4 random samples
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(self.y.shape)
        #the output is an array of zeros with the shape of y

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
    #note that for simplicity we have assumed the biases to be zero

    def backprop(self):
        # used the chain rule to find the derivative of the loss function (sum of squares)
        #wrt weights and biases
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                              self.weights2.T) * sigmoid_derivative(self.layer1)))

    #update the weights with the derivative of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

if __name__ == "__main__":
    #here we make our data to learn with
    X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
    y = np.array([[0],[1],[1],[0]])
    nn = NeuralNetwork(X,y)

    for i in range(2000): #1500 iterations
        nn.feedforward()
        nn.backprop()

    print(nn.output)

    #the output will try to estimate y by iterating 2000 times the feedforward and backprop