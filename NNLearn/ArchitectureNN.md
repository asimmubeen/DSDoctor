# Architecture of a Neural Network

A neural network consists of interconnected "neurons" arranged in "layers." Each neuron takes one or more inputs, performs a simple calculation, and produces an output, which is then passed on to the next layer.

The simplest type of neural network is a feedforward neural network, which consists of an input layer, one or more hidden layers, and an output layer. The input layer receives the input data, and each neuron in the layer corresponds to a single input feature. The hidden layers perform intermediate computations on the input data, and each neuron in the hidden layers takes as input the output of all the neurons in the previous layer. Finally, the output layer produces the final output of the neural network, which corresponds to the prediction or classification of the input data.

<p align="center"> <img src="https://user-images.githubusercontent.com/24811295/221954806-55f5812c-da4c-4393-a0ba-26d41ff603ba.png" height="500" width="600"> </p>

Each neuron in a neural network performs a simple calculation, typically a weighted sum of its inputs followed by a non-linear activation function. The weights determine the strength of the connections between neurons, and the activation function introduces non-linearity into the network, allowing it to model more complex relationships in the input data.

During training, the neural network learns to adjust its weights and biases in order to minimize the difference between its predicted output and the true output. This is achieved through an optimization algorithm, such as stochastic gradient descent, which iteratively adjusts the weights and biases of the network based on the error between the predicted output and the true output.

Overall, the architecture of a neural network is relatively simple: it consists of interconnected neurons arranged in layers, with each neuron performing a simple computation on its inputs. However, the complexity of the network arises from the large number of neurons and connections, as well as the non-linear activation functions and optimization algorithms used to train the network.

## Neural Network Layers
Neural networks can have different numbers of layers depending on the complexity of the problem they are solving. The most common types of layers used in neural networks are:

### Input layer
This layer receives the input data, which is then processed by the following layers. The number of nodes in the input layer corresponds to the number of input features in the data.

### Hidden layer
This layer is sandwiched between the input and output layers and contains multiple neurons that perform complex computations on the input data. The number of hidden layers and the number of neurons in each layer can vary depending on the problem being solved.

### Output layer
This layer produces the final output of the neural network, which can be a single value (in the case of regression) or a set of probabilities (in the case of classification). The number of nodes in the output layer corresponds to the number of output values required.

Each layer in a neural network can use different types of activation functions to transform the input data. Some commonly used activation functions are:

### Sigmoid function
This function maps any input value to a value between 0 and 1, making it useful for binary classification problems.

### ReLU function
This function sets all negative values in the input data to zero and leaves positive values unchanged, making it useful for image recognition problems.

### Tanh function
This function maps any input value to a value between -1 and 1, making it useful for problems where the input data has negative and positive values.

There are also other activation functions like softmax, Leaky ReLU, and Swish that can be used depending on the problem and the desired output.

Click [here](/NNLearn/IntrotoNN.md) to go back ...



