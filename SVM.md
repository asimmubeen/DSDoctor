# Support Vector Machines (SVM)

Support Vector Machines (SVM) is a type of supervised machine learning algorithm used for classification and regression analysis. SVM is based on the idea of finding a hyperplane that can best separate the data into different classes. The hyperplane is a decision boundary that separates the data into two classes, such that the margin between the classes is maximized. The margin is defined as the distance between the hyperplane and the closest data points, which are known as support vectors.

SVM can handle both linear and non-linear data and can be used for both binary and multi-class classification problems. The algorithm works by transforming the input data into a higher-dimensional feature space, where a linear or non-linear boundary can be found. In the case of a linear boundary, a simple hyperplane is used to separate the data into two classes. In the case of a non-linear boundary, a technique known as kernel trick is used to map the input data into a higher-dimensional space, where a linear boundary can be found.

SVM can also be used for regression analysis by finding a hyperplane that fits the data such that the error between the prediction and the actual data is minimized. In this case, the margin is defined as the distance between the hyperplane and the closest data points, which are again called support vectors.

SVM is a powerful and versatile machine learning algorithm that can handle both linear and non-linear data, and is commonly used in a variety of applications, such as text classification, image classification, and bioinformatics.


## Mathematical interpretation

The mathematical explanation of SVM is based on the concept of finding a hyperplane that best separates the data into different classes. The hyperplane is represented by a linear equation of the form:

$$w^Tx + b = 0$$,

where $w$ is the normal vector to the hyperplane and $b$ is the bias term. The goal of SVM is to find the values of w and b such that the hyperplane separates the data into different classes with the maximum margin, which is defined as the distance between the hyperplane and the closest data points from either class. These closest data points are called support vectors.

In order to find the hyperplane that maximizes the margin, SVM uses the optimization problem known as the primal problem. The primal problem can be expressed as the following optimization problem:

$$min \frac{1}{2} |w|^2$$

subject to $$y_i(w^T x_i + b) ≥ 1, i = 1, 2, ..., n$$

where $n$ is the number of data points, $x_i$ and $y_i$ are the input data and corresponding labels, respectively. The optimization problem can be solved using a variety of optimization algorithms, such as gradient descent or quadratic programming.

In the case of non-linear data, SVM uses the technique known as kernel trick to map the input data into a higher-dimensional feature space, where a linear hyperplane can be found. The optimization problem in this case is known as the dual problem, and it can be expressed as the following optimization problem:

$$max α^T t$$

subject to $$α_i ≥ 0, i = 1, 2, ..., n$$

and $$α_i y_i = 0, i = 1, 2, ..., n$$

where $α$ is the vector of Lagrange multipliers and t is the vector of the target values. The optimization problem can be solved using a variety of optimization algorithms, such as the Sequential Minimal Optimization (SMO) algorithm.

SVM uses the mathematical concepts of hyperplanes and optimization problems to find the boundary that separates the data into different classes with the maximum margin.

## Python Code
A simple example of coding Support Vector Machines (SVM) in Python using the scikit-learn library:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm

# Load the iris dataset
iris = datasets.load_iris()
X = iris["data"]
y = iris["target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train a SVM classifier on the training data
clf = svm.SVC(kernel='linear', C=1, random_state=0)
clf.fit(X_train, y_train)

# Predict the class labels for the test data
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVM (linear kernel) Decision Boundary')
plt.show()
```
In this example, we use the scikit-learn library to load the iris dataset, split it into training and testing sets, and train a SVM classifier with a linear kernel. The classifier is then used to predict the class labels for the test data, and the accuracy is evaluated. Finally, the decision boundary is plotted to visualize the separation of the classes by the hyperplane found by the SVM algorithm.

