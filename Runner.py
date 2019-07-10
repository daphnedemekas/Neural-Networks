

import MNIST
from MNIST import *

import Number_Guess


net = Number_Guess.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
