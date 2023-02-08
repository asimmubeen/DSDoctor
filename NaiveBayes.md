# Naïve Bayes

Naive Bayes is a probabilistic machine learning algorithm based on Bayes' Theorem, which states that the probability of an event (in case of classification, a class label) given certain information (the feature values of an instance) can be calculated as the product of the prior probability of the event and the likelihood of the information given the event, divided by the marginal probability usually called as the evidence.

In the context of Naive Bayes, the algorithm is "naive" because it assumes that all the features are independent, which is often not true and is a "naive" assumption. However, despite this oversimplification, Naive Bayes algorithms are known to work well in practice, especially for high-dimensional datasets with a large number of features. 

**Note**: One can designed an informed Bayes algorithm based on the available information using proper prior and informed likelihood model.

There are three main variants of Naive Bayes:

**Gaussian Naive Bayes**: This assumes that the continuous features follow a normal / Gaussian distribution.

**Multinomial Naive Bayes**: This is used for discrete data such as text, where the features represent the frequency of occurrences of words.

**Bernoulli Naive Bayes**: This is similar to Multinomial Naive Bayes, but it considers binary occurrence or non-occurrence of features.

Naive Bayes is often used for text classification and spam filtering, as well as for sentiment analysis and **disease diagnosis**. Due to its simplicity and speed, Naive Bayes can be a good choice for large datasets, especially when the goal is to quickly build a baseline model.

## Mathematical interpretation

Naive Bayes is based on Bayes' theorem, which states that the probability of a class $C$ given some observed features $X$ can be calculated as:

$$P(C | X) = \frac{P(X | C) * P(C)}{P(X)}$$

where $P(C | X)$ is the posterior probability of class $C$ given the features $X$, $P(X | C)$ is the likelihood of the features $X$ given class $C$, $P(C)$ is the prior probability of class $C$, and $P(X)$ is the marginal probability of the features $X$.

In the case of Naive Bayes, the features are assumed to be conditionally independent given the class, so the likelihood term can be written as a product of individual  probabilities of all the features:

$$P(X | C) = P(x_1 | C) * P(x_2 | C) * ... * P(x_n | C)$$

The prior probabilities can be estimated from the training data, and the likelihoods can be estimated or assumed using either Gaussian, Multinomial, or Bernoulli distributions, depending on the type of Naive Bayes algorithm being used.

<p align="center"> <img src ="https://user-images.githubusercontent.com/24811295/217283810-ce35567a-aa44-4c8b-9500-0033d3cb3d9a.png" height=250 width=650> </p>

<p align="center"> Image Adopted From: https://thatware.co/naive-bayes/</p>


Given these estimates, the posterior probability of each class can be calculated for a new instance, and the class with the highest probability can be chosen as the prediction.

## Python Code

Here's a simple example of how to implement Gaussian Naive Bayes in Python using the scikit-learn library:

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification

# Generate a sample dataset for demonstration purposes
X, y = make_classification(n_samples=1000, n_features=4, random_state=0)

# Initialize the classifier
gnb = GaussianNB()

# Fit the classifier to the training data
gnb.fit(X, y)

# Predict the class labels for a sample input
sample_input = np.array([[0, 0, 0, 0]])
prediction = gnb.predict(sample_input)
print("Prediction for sample input:", prediction)

```

This example generates a sample dataset using the make_classification function from scikit-learn and fits a Gaussian Naive Bayes classifier to the generated data. Finally, the classifier is used to make a prediction for a sample input.

Similarly, _**Multinomial Naive Bayes**_ and **_Bernoulli Naive Bayes_** can be implemented using the **MultinomialNB** and **BernoulliNB** classes from the scikit-learn library, respectively.
