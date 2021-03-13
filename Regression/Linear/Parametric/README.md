# Parametric Linear Regression

Based on notes from lecture 2 of the course [Stanford CS229: Machine Learning | Autumn 2018](https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) with professor Andrew Y. Ng, I implemented and explored **Batch Gradient Descent**, **Stochastic Gradient Descent** and **Evaluation Through Normal Equation** algorithms using Python, three different approach based on gradient descent to calculate linear regression.

### Dependencies 
[matplotlib](https://github.com/matplotlib/)

[numpy](https://github.com/numpy)

## Batch Gradient Descent
The formula used for this first approach was derived from the notes and is the following:

<p align="center"><img src="http://www.sciweavers.org/download/Tex2Img_1615042970.png"></p>

Where θj, the parameters of a linear combination (otherwise known as weights), is subsequently updated based on a sum of an "actual value" y^(i) minus a "predicted value" hθ(x^(i)) times a particular feature xj^(i) weighted by a learning rate α. Iteration occurs until θj converge. The term with Σ is the gradient of a cost function J(θ).


The snippet of code that implements the aforementioned formula is defined as:

```python
def batch_gradient_descent(self):
        for j in range(len(self.theta)):
            for i in range(self.m):
                self.theta[j] = self.theta[j] + \
                    (self.learning_rate * (np.sum(self.y) - np.sum(self.X * self.theta)) * self.X[i][j])
                self.retrieved_thetas.append(self.theta.copy()) # Isn't part of the formula
```

In this approach, each value of θj is evaluated individually at each *j-th* iteration. The number of iterations is defined by the quantity of parameters θ (expressed in the outer for loop) and the number of instance of the training data represented as *m* (expressed in the inner for loop).

## Stochastic Gradient Descent
The formula used for this second approach was also derived from the notes and is defined as follow:

<p align="center"><img src="http://www.sciweavers.org/download/Tex2Img_1615044029.png"></p>

The main difference is that, when updating θj, only one data point from the training set is consider in the calculation at each iteration, whereas in Batch Gradient Descent, every data point was added up at each iteration. One might consider the stochastic approach when dealing with huge data sets, because in a scenario like this, Batch Gradient Descent becomes inviable due to the amount of time the algorithm would consume.

The snippet of code that implements this second approach is the following:

```python
def stochastic_gradient_descent(self):
        FLAG = 1
        NUM_STEPS = 1000
        steps = [0] * len(self.theta)

        while(FLAG and NUM_STEPS):
            for i in range(self.m):
                for j in range(len(self.theta)):
                    step = self.learning_rate * (self.y[i] - self.__hypothesis(self.X[i])) * self.X[i][j]
                    self.theta[j] = self.theta[j] + step
                    steps[j] = step

                self.retrieved_thetas.append(self.theta.copy()) # Isn't part of the formula

                if all(abs(steps[i]) < 0.0001 for i in range(len(steps))):
                    FLAG = 0
                NUM_STEPS -=1
```

My idea is to explore different implementations of gradient descent. In contrast with the batch one, in this case I chose to update every θj at the same *i-th* iteration, so they are evaluated almost at the same time. The condition defined to stop the iteration was to verify if every step taken in each calculation of the parameters were less than 0.0001 or if a maximum number 1000 of steps were reached.

## Normal Equation
The idea behind this approach is to set ∇J(θ) to 0, because the local minimum is a point where the value of gradient is zero. After working algebraically from this, we reach the following formula:

<p align="center"><img src="http://www.sciweavers.org/download/Tex2Img_1615376864.png"></p>

This method doesn't rely on any iteration, evaluation of parameters' values occur executing a single operation.

The snippet of code that implements this third approach is the following:

```python
def normal_equation(self):
        y = self.y[:, np.newaxis] # isn't part of the formula
        
        theta = ((inv(self.X.T@self.X)@self.X.T@y).flatten())

        return np.array([theta])
```

## Visual examples of parameters' calculation and its effect in the linear function
The animation and the 3D plotting were made using [matplotlib](https://github.com/matplotlib/). I wrote a script that, given a list of training data X, y and a list of all calculated θ to an instantiated object, it's possible to generate an animation of changes in the linear function. This script is named ```plot2D.py``` in this repository.

### Batch Gradient Descent
<p align="center"><img src="https://user-images.githubusercontent.com/29299799/110643351-c5487300-8192-11eb-978b-9f441364457f.gif"></p>

The velocity of calculation doesn't match the real time taken. In reality, with this amount of data, calculation is practically instantaneous. Nonetheless, θj being evaluated one at a time is accurate, it was implemented this way to demonstrate different flavors of evaluation.

### Stochastic Gradient Descent
<p align="center"><img src="https://user-images.githubusercontent.com/29299799/110643352-c5e10980-8192-11eb-85ba-398fc08e52d0.gif"></p>

As clarified in the previous section, the velocity in which θj is being evaluated is not real. But, unlike the other approach, they are being calculated at the same time.

The flickering near the end is due to the algorithm's stochastic nature. That is, because the steps towards the point where ∇J(θ)=0 is calculated based upon only one sample point at each iteration, the optimum minimum is never reached, rather steps are taken around it, but they never converge into it.

### Normal Equation
<p align="center"><img src="https://user-images.githubusercontent.com/29299799/110643347-c4afdc80-8192-11eb-94f2-192884d53e4e.png"></p>

This method returns only one set of thetas which are already suited.

### Data with multiple features
The two previous examples dealt with *x* having only one feature, therefore, the hypothetical linear function had the form

<p align="center"><img src="http://www.sciweavers.org/download/Tex2Img_1615045196.png"></p>

The function expressed a 2-dimensional geometry. But, those same algorithms can also deal with more features, therefore more dimensions. In the following case, with a dummy training dataset, the hypothetical function is defined as

<p align="center"><img src="http://www.sciweavers.org/download/Tex2Img_1615045238.png"></p>

It can model data dispersed in a 3-dimensional line, but the same parameters also define a plane:

<div align="center">
        <img src="https://user-images.githubusercontent.com/29299799/110643241-a518b400-8192-11eb-8730-6af7e448a996.png" width="40%" height="40%" style="float:left;">
        <img src="https://user-images.githubusercontent.com/29299799/110643248-a8ac3b00-8192-11eb-93d9-94fc79fe98ba.png" width="40%" height="40%" style="float:left;">
</div>
