Based on notes from lecture 2 of the course [Stanford CS229: Machine Learning | Autumn 2018](https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) with professor Andrew Y. Ng, Batch Gradient Descent algorithm and Stochastic Gradient Descent algorithm, two different approach based on gradient descent to calculate linear regression, were implemented and explored using Python.

## Batch Gradient Descent

 The formula used for this first approach was derived from the notes and is the following:
$$
\theta_{j}:=\theta_{j}+\alpha\sum_{i=1}^m(y^{(i)}-h_{\theta}(x^{(i)}))x_{j}^{(i)}
$$
Where $\theta_j$, the parameters of a linear combination (otherwise known as weights), is subsequently updated based on a sum of an "actual value" $$y^{(i)}$$ minus a "predicted value" $$h_\theta(x^{(i)})$$  times a particular feature $$x_{j}^{(i)}$$ weighted by a learning rate $$\alpha$$. Iteration occurs until $$\theta_j$$ converge. The term $$\sum_{i=1}^m(y^{(i)}-h_{\theta}(x^{(i)}))x_{j}^{(i)}$$ is the gradient of a cost function $$J(\theta)$$.

The snippet of code that implements the aforementioned formula is defined as:

```python
def batch_gradient_descent(self):
        for j in range(len(self.theta)):
            for i in range(self.m):
                self.theta[j] = self.theta[j] + (self.learning_rate * (np.sum(self.y) - np.sum(self.X * self.theta)) * self.X[i][j])
                self.retrieved_thetas.append(self.theta.copy())
```

In this approach, each value of $$\theta_j$$ is evaluated individually at each *j-th* iteration. The number of iterations is defined by the quantity of parameters $$\theta$$ (expressed in the outer for loop) and the number of instance of the training data represented as $$m$$ (expressed in the inner for loop).

## Stochastic Gradient Descent

The formula used for this second approach was also derived from the notes and is defined as follow:
$$
\theta_{j}:=\theta_{j}+\alpha(y^{(i)}-h_{\theta}(x^{(i)}))x_{j}^{(i)}
$$
The main difference is that, when updating $$\theta_j$$ only one data point from the training set is consider in the calculation at each iteration, whereas in Batch Gradient Descent, every data point was added up at each iteration. One might consider the stochastic approach when dealing with huge data sets, because in a scenario like this, Batch Gradient Descent becomes inviable due to the amount of time the algorithm would consume.

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

                self.retrieved_thetas.append(self.theta.copy())

                if all(abs(steps[i]) < 0.0001 for i in range(len(steps))):
                    FLAG = 0
                NUM_STEPS -=1
```

My idea is to explore different implementations of gradient descent. In contrast with the batch one, in this scenario I chose to update every $$\theta_j$$ at the same *i-th* iteration, so they are evaluated almost at the same time. The condition defined to stop the iteration was to verify if every steps taken in each calculation of the parameters were less than $$0.0001$$ or if a maximum number $$1000$$ of steps were reached.