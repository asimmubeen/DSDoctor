# Generative Models
Generative models in machine learning are a class of models that can learn the distribution of a set of data and then generate new, similar samples from that learned distribution. The idea is to learn the underlying structure of the data in such a way that the model can generate new, unseen data that is representative of the original data.

Generative models can be used for a variety of tasks, such as image synthesis, text generation, anomaly detection, and more. Some examples of generative models include Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and Autoregressive Models (ARMs).

In [GANs](#gans), two neural networks, a generator and a discriminator, are trained together in an adversarial manner. The generator network learns to generate new data samples that resemble the training data, while the discriminator network tries to distinguish the generated samples from the real samples.

VAEs are a type of generative model that use an encoder-decoder architecture. The encoder network maps the input data to a lower-dimensional representation, while the decoder network maps the lower-dimensional representation back to the original data space. The goal is to learn a compact representation of the data that captures the most important features, and use this representation to generate new samples.

ARMs are a type of generative model that use a conditional probability to generate new samples. The model learns the dependencies between different elements of the data, and uses this information to generate new samples by sampling from the learned conditional probability distribution.

Generative models can be a powerful tool for creating new data samples that are representative of the training data, and can be used for a wide range of applications in machine learning and artificial intelligence.

# GANs
Generative Adversarial Networks (GANs) are a deep learning architecture that consist of two main components: a generator network and a discriminator network. The goal of GANs is to generate new data samples that are similar to a given training dataset.

The generator network is responsible for generating new samples, while the discriminator network is responsible for evaluating the generated samples and determining if they are similar to the real, training data. The two networks are trained together in an adversarial manner, where the generator network tries to generate samples that can fool the discriminator network, and the discriminator network tries to accurately distinguish the generated samples from the real samples.

The training process of a GAN can be summarized as follows:

1. The generator network generates a new sample, which is then passed to the discriminator network for evaluation.

2. The discriminator network evaluates the generated sample and outputs a probability indicating how similar the sample is to the real, training data.

3. The generator network uses the feedback from the discriminator network to update its parameters and generate a new sample.

4. The process continues until the generator network can generate samples that are indistinguishable from the real, training data, or the discriminator network can perfectly distinguish the generated samples from the real samples.

GANs have been used for a variety of tasks, including image synthesis, style transfer, and image completion, among others. Despite their success, GANs can be difficult to train and often suffer from instability issues such as mode collapse, where the generator network generates only a limited number of different samples. Nonetheless, GANs remain a powerful tool for generating new data samples that are representative of a given training dataset

Let's assume that we have a real data distribution represented by $p_data(x)$, where x is a data point in a high-dimensional space. The goal of the generator network is to generate new samples from a noise distribution $p_z(z)$, where $z$ is a low-dimensional random noise vector. The generator network is represented as a function $G(z; θ_g)$, where $θ_g$ are the parameters of the generator network. The generated data distribution is represented by:

$$p_g(x) = p_z(z) * G(z; θ_g)$$ where $*$ represents the convolution operation.

The discriminator network is represented as a function $D(x; θ_d)$, where $θ_d$ are the parameters of the discriminator network. The discriminator network outputs a scalar value between $0$ and $1$, indicating the probability that a given sample x came from the real data distribution $p_{data}(x)$ rather than the generated data distribution $p_g(x)$.

The GAN training process can be formalized as a two-player minimax game between the generator and the discriminator, where the generator tries to generate samples that can fool the discriminator, and the discriminator tries to accurately distinguish the generated samples from the real samples. The objective function for the GAN can be expressed as:

$$J(θ_g, θ_d) = min_{θ_g} max_{θ_d} E_{x}p_{data(x)}{log D(x;θ_d)} + E_{z}p_{z(z)}{log(1 - D(G(z; θ_g); θ_d))}$$

The objective function is optimized by alternating between updating the parameters of the generator network θ_g and the parameters of the discriminator network $θ_d$. The generator network is updated to generate samples that maximize the probability of the discriminator making a mistake, while the discriminator network is updated to minimize the probability of making a mistake.

In practice, the optimization of the objective function can be challenging, due to the instability of the minimax game and the possible collapse of the generated data distribution to a limited number of modes. Nevertheless, GANs have shown to be a powerful tool for generating new data samples that resemble a given training dataset, and continue to be an active area of research in deep learning.

## Python Code (TensorFlow)
Here's an example of a Generative Adversarial Network (GAN) implemented using TensorFlow in Python:

```python
import tensorflow as tf
from tensorflow.keras import layers

# Define the generator
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(512, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(28*28*1, use_bias=False, activation='tanh'))
    model.add(layers.Reshape((28, 28, 1)))

    return model

# Define the discriminator
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(1))

    return model

# Define the loss functions for the generator and discriminator
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Define the optimizer for the generator and discriminator
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Define the GAN
generator = make_generator_model()
discriminator = make_discriminator_model()

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Train the GAN
EPOCHS = 100
BATCH_SIZE = 256

for epoch in range(EPOCHS):
    for batch in range(train_images.shape[0] // BATCH_SIZE):
        images = train_images[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
        train_step(images)
```
This code defines a GAN that generates images similar to those in the MNIST dataset. The generator creates images from random noise, and the discriminator tries to distinguish between real and fake images. The generator is trained to create images that fool the discriminator, while the discriminator is trained to correctly identify real and fake images.

## Python Code (Keras)
Here's an example of a simple GAN implemented in Python using the **Keras API**:

```python
import numpy as np
from keras.layers import Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load MNIST data
from keras.datasets import mnist
(x_train, _), (_, _) = mnist.load_data()

# Rescale -1 to 1
x_train = x_train / 127.5 - 1.
x_train = np.expand_dims(x_train, axis=3)

# Generator model
generator = Sequential()
generator.add(Dense(128 * 7 * 7, activation="relu", input_dim=100))
generator.add(Reshape((7, 7, 128)))
generator.add(LeakyReLU(alpha=0.01))
generator.add(Dense(256, activation="relu"))
generator.add(LeakyReLU(alpha=0.01))
generator.add(Dense(1, activation="tanh"))
generator.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.5))

# Discriminator model
discriminator = Sequential()
discriminator.add(Flatten(input_shape=(28, 28, 1)))
discriminator.add(Dense(256, activation="relu"))
discriminator.add(LeakyReLU(alpha=0.01))
discriminator.add(Dense(1, activation="sigmoid"))
discriminator.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.5))

# Combined model
gan = Sequential()
gan.add(generator)
gan.add(discriminator)
gan.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.5))

# Training loop
batch_size = 128
epochs = 10000
sample_interval = 1000
for epoch in range(epochs):
    # Train discriminator
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_imgs = x_train[idx]
    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_imgs = generator.predict(noise)
    d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # Train generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    
    # Print progress
    if epoch % sample_interval == 0:
        print(f"Epoch {epoch}: Discriminator loss: {d_loss}, Generator loss: {g_loss}")
        noise = np.random.normal(0, 1, (16, 100))
        gen_imgs = generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(4, 4)
        count = 0
        for i in range(4):
            for j in range(4):
                axs[i,j].imshow(gen_imgs[count, :,:,0], cmap="gray")
                axs[i,j].axis("off")
                count += 1
        plt.show()
```

This code trains a GAN on the MNIST dataset to generate handwritten digits. The generator model takes a random noise vector as input and produces an image, while the discriminator model takes an image as input and outputs a probability
