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

$$J(θ_g, θ_d) = min_{θg} max_{θd} E_{xp_data(x)}{log D(x;θ_d)} + E_{zpz(z)}{log(1 - D(G(z; θ_g); θ_d))}$$

The objective function is optimized by alternating between updating the parameters of the generator network θ_g and the parameters of the discriminator network $θ_d$. The generator network is updated to generate samples that maximize the probability of the discriminator making a mistake, while the discriminator network is updated to minimize the probability of making a mistake.

In practice, the optimization of the objective function can be challenging, due to the instability of the minimax game and the possible collapse of the generated data distribution to a limited number of modes. Nevertheless, GANs have shown to be a powerful tool for generating new data samples that resemble a given training dataset, and continue to be an active area of research in deep learning.
