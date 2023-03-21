# Neural Netwrok Application
Application of any algorithm is as important as to understand it. In this example we will learn how to use a neural network to classify images of handwritten digits using Python and the popular deep learning library, Keras:

## Python Code

Step 1: Import necessary libraries and load the dataset

```python
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Flatten the images into a 1D array of 784 features
train_images = train_images.reshape((60000, 784))
test_images = test_images.reshape((10000, 784))

# Convert the labels to categorical format
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
```

Step 2: Define the neural network architecture

```python
from keras import models
from keras import layers

# Define the neural network architecture
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(784,)))
model.add(layers.Dense(10, activation='softmax'))
```
Step 3: Compile the model

```python
# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

Step 4: Train the model on the training data

```python
# Train the model on the training data
model.fit(train_images, train_labels, epochs=5, batch_size=128)
```

Step 5: Evaluate the model on the test data

```python
# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
```

This code defines a neural network with a single hidden layer of 512 neurons and an output layer of 10 neurons (corresponding to the 10 possible digits). The model is then compiled with the RMSprop optimizer and categorical cross-entropy loss function, and trained on the MNIST dataset for 5 epochs with a batch size of 128. Finally, the model is evaluated on the test set and the accuracy is printed.

Click [here](/NNLearn/IntrotoNN.md) to go back ...
