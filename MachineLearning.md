# Machine Learning
Machine learning is a subfield of artificial intelligence that allows systems to automatically improve their performance on a specific task through experience. There are three main types of machine learning:

1. **Supervised learning**
2. **Unsupervised learning**
3. **Reinforcement learning**

### **Supervised learning** 
In this type of learning, the algorithm is trained on a labeled dataset, meaning that the data includes both the inputs (features) and the desired outputs (labels). The goal is to learn a mapping from inputs to outputs, such that the algorithm can make accurate predictions on new, unseen data. 

Some common supervised machine learning algorithms include:

**[Linear Regression](/LinearRegression.md)**: used to model the linear relationship between a dependent variable and one or more independent variables.

**[Logistic regression](/LogisticRegression.md)**: used to model the probability of a binary outcome.

**[Support Vector Machines (SVM)](/SVM.md)**: used for classification and regression by finding the hyperplane that best separates the data into classes.

**[Decision Trees](/DecisionTrees.md)**: used to model decisions or decisions based on certain conditions.

**[Random Forest](/RandomForest.md)**: an ensemble of decision trees, used to improve the accuracy and stability of the predictions.

**[Naive Bayes](/NaiveBayes.md)**: used for classification by making predictions based on the probability of each class and the likelihood of the features given the class.

**K-Nearest Neighbors (KNN)**: used for classification by finding the K nearest data points and returning the majority class.


Neural Networks: used for a wide range of tasks, including image recognition and natural language processing, by modeling complex relationships between inputs and outputs.

### **Unsupervised learning**
In unsupervised learning, the algorithm is trained on an unlabeled dataset, meaning that the data includes only the inputs (features) and not the desired outputs (labels). The goal is to identify patterns or relationships in the data without being told what the outputs should be. Examples: clustering (e.g. k-means), dimensionality reduction (e.g. principal component analysis).

Unsupervised machine learning is a type of machine learning where the goal is to find structure or patterns in the data, without being given explicit labels or target values. Some common unsupervised learning methods are:

**[Clustering](/Clustering.md)**: This method groups similar data points together based on some similarity metric. Examples of clustering algorithms include K-Means, Hierarchical Clustering, and DBSCAN.

**Dimensionality Reduction**: This is the process of reducing the number of features in the data while preserving the important information. This can be useful for visualizing high-dimensional data, reducing the complexity of the data, and removing noise. Common dimensionality reduction techniques include Principal Component Analysis (PCA), t-SNE, and Autoencoders.

**Anomaly detection**: This method is used to detect data points that are significantly different from the majority of the data. Examples of anomaly detection algorithms include Isolation Forest, Z-Score, Mahalanobis Distance, One-Class SVM, and Local Outlier Factor.

**Association Rule Learning**: This is the process of finding relationships between variables in a large dataset. Association rule learning algorithms are commonly used in market basket analysis and recommender systems. Apriori and Eclat are two common association rule learning algorithms.

**[Generative models](/GenerativeModels.md)**: These models generate new data samples that are similar to the input data. Examples of generative models include Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and Restricted Boltzmann Machines (RBMs).

**Non-negative Matrix Factorization (NMF)**: This is the process of factorizing a matrix into two non-negative matrices, where the product of the two matrices approximates the original matrix. NMF is commonly used for topic modeling and dimensionality reduction.

These unsupervised learning methods can be used for a wide range of tasks, such as data visualization, data compression, data generation, data pre-processing, and anomaly detection.


### **Reinforcement learning**
In reinforcement learning, an agent learns to interact with an environment by performing actions and observing the resulting rewards or penalties. The goal is to learn a policy that maximizes a reward signal over time. Examples: playing chess or Go, autonomous navigation.
