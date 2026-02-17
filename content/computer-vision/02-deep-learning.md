---
title: Deep Learning
---

Traditional CV techniques were always based on hand-crafted algorithms. After the ML methods began gaining traction, they appeared tries of applying the to computer vision tasks. Most common ML task defined in CV is **classification** (supervised learning), in which we are given a set of input values $\{\mathbf{x}_i \}$ (often being some feature vectors or entire images) paired with corresponding class labels $\{t_i\}$, which come from predefined set of classes $\{c_i\}$. Despite the dominance of supervised learning, we can also describe some methods for unsupervised or semi-supervised learning in CV.

## Supervised Learning

One of the most common applications of classification in CV is semantic **image classification**, where we simply label the entire image with some class. While tackling this task, we often don't have the access to true probability distribution over the inputs (moreover the join probability of outputs given inputs). Therefore, we will use trainig data distribution as proxy for real-world distribution. This is known as _empirical risk minimization_, where the expected risk is estimated with:

$$
    E(w) = \frac{1}{N} \sum L(y_i, f(x_i, w)),
$$

where $L$ measures loss of predicting an output $f(x_i, w)$ for input $x_i$ and model parameters $w$ when the expected label is $y_i$.

### Preprocessing

It is often a good idea to prepare the data for classification. The most common preprocessing steps are:

- centering - subtracting mean from features.
- standarizing - scaling the feature so its variance is equal to $1$.
- whitening - computing SVD and rotating the feature space so the final dimensions are uncorrelated and have unit variance.

### Nearest Neighbors

Very simple "brute-force" method. We take the closest $k$ neighbors for a given input data and return the label that occures the most. We can imagine that low numbers of $k$ makes the method behave more abuptly (we only sample few neighbors) resulting in overfitting, and when $k$ gets large it is prone to underfitting.

### Bayesian Classification

If we are able to come up with analytic model of feature construction and noising, or if we can gather enought samples, we can determine the probability distributions of the feature vectors for each class $p(x | c_i)$ as well as prior class likelihoods $p(c_i)$. According to Bayes' rule, the likelihood of $c_i$ given $x$ is given by:

$$
    p_i = p(c_i | x) = \frac{p(x | c_i)p(c_i)}{\sum p(x | c_j) p(c_j)} = \frac{\exp(l_i)}{\sum \exp(l_j)},
$$

where the second for is known as _normalized exponential_ or a **softmax**. The quantity:

$$
    l_k = \log p(x | c_k) + \log p(c_k)
$$

is the _log-likelihood_ of a sample $x$ being from class $c_k$. The process of applying formula PLACEHOLDER to find the likelihood of class $c_k$ given $x$ is known as **Bayesian classification**. In case the components of the feature are strongly independent i.e.

$$
    p(x|c_k) = \prod_i p(x_i | c_k).
$$

the resulting technique is called **Naive Bayes Classifier**. For binary classification task, we can rewrite PLACEHOLDER as

$$
    p(c_0 | x) = \frac{1}{1 + \exp(-l)} = \sigma(l),
$$

where $l = l_0 - l_1$ is the difference between the two class log-likelihood and is known as _logit_.

---

Papers:

$$
$$
