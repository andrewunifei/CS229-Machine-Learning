# General Linear Model
Based on notes from lecture 4 of the course [Stanford CS229: Machine Learning | Autumn 2018](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) with professor Andrew Y. Ng, I wrote the following text exploring the concept of **General Linear Model**.

## The exponential family
In order to explore the ideia of General Linear Model (GLM), it is first necessary to describe the notion of exponential family distributions. The exponential family is constituted by members. A member of the exponential family is a class distributions that has a structure defined as:

<p align="center"><img src="https://user-images.githubusercontent.com/29299799/113068103-fe717300-9193-11eb-8d59-13c476c4eb2d.png"></p>

Where η is the **natural parameter** or **canonical parameter**, T(y) is the **sufficient statistic**, a(η) is the **log partition function** and exp(-a(η)) has a function of **normalization**. Choosing T, a and b determine a family of distributions parameterized by η.

Examples of members of the exponential family are:

* **Gaussian Distributions**

  Define a normal distribution. Suited to derive linear regression. 
  
* **Bernoulli Distribution**

  Define a discrete probability distribution in the space {0, 1}. Suited to derive logistic regression (binomial).
 
* **Multinomial Distribution**

  Define a discrete probability distribution generalizing the binomial distribution. Suited to derive multinomial linear regression.

* **Poisson Distribution**

  Suited for modelling count-data.

* **Gamma Distribution**
* **Exponential Distribution**
   
  Both suited for modelling countinuous non-negative random variables, like time intervals.
  
* **Beta Distribution**
* **Dirichlet Distribution**

  Both suited for distributions over probabilities.

There are others members that belongs to the exponential family.

## General Linear Model
When dealing with a regression or classification problem, where one needs to predict the value of a particular random variable y given a set of features x, it is necessary to come up with a model. In order to develop a General Linear Model for a problem like this, three design choices must be made:

1. The distribution of y, given x and set of parameters θ, is a member of the exponential family.
2. In these types of problems, T(y) = y, therefore the objetive is to predict T(y) given x.
3. There is a linear relation between η and set of features x expressed as η = θ^Tx.

Generally a General Linear Model learning algorithm will have the following structure:

<p align="center"><img src="https://user-images.githubusercontent.com/29299799/113524827-ae295500-9587-11eb-95eb-3e332de70d32.png"></p>

Differing only the **hypothesis function** h(x).
