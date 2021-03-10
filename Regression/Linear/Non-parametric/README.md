# Non-parametric Linear Regression

Based on notes from lecture 3 of the course [Stanford CS229: Machine Learning | Autumn 2018](https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) with professor Andrew Y. Ng, I implemented and explored **Locally Weighted Regression** using Python to calculate linear regression in a non-linear function.


### Dependencies 
[matplotlib](https://github.com/matplotlib/)

[numpy](https://github.com/numpy)

## Locally Weighted Regression
Locally Weighted Regression is the task of modeling a linear function to a particular value of x considering its neighbors values. This consideration is done by applying a weight to all x^(i) values in the dataset. Those near x are weighted more heavily, whereas those far away are weighted more softly, therefore the former has a greater impact in the final calculation.

The formula used to implement this algorithm is defined as:

<p align="center"><img src="http://www.sciweavers.org/download/Tex2Img_1615380331.png"></p>

Where W is defined as:

<p align="center"><img src="http://www.sciweavers.org/download/Tex2Img_1615381125.png"></p>

And w^(i) as:

<p align="center"><img src="http://www.sciweavers.org/download/Tex2Img_1615381386.png"></p>

The snippet of code that implements those formulas is as follows:

```python
    def __w(self, i, x, tau):
        '''
        Definition of w function.
        Cost values will be weighted by the return of this function
        '''
        return np.exp(-(np.sum((self.X[i] - x)**2))/(2*tau**2))

    def locally_weighted(self, x):
        x = np.r_[1, x]
        y = self.y[:, np.newaxis]
        W = np.zeros((self.m, self.m))

        np.fill_diagonal(W, list(0.5 * self.__w(i, x, self.tau) for i in range(self.m)))
        
        thetas = (inv(self.X.T@W@self.X)@self.X.T@W@y)

        return thetas.T@x.T # Prediction on given x 
```

## Visual representation
<div align="center">
<img src="https://github.com/andrewunifei/CS229-Machine-Learning/blob/main/Regression/Linear/Non-parametric/Resources/Tau/a-tau_10.png" width="40%" height="40%" style="float:left">
<img src="https://github.com/andrewunifei/CS229-Machine-Learning/blob/main/Regression/Linear/Non-parametric/Resources/Tau/b-tau_1.png" width="40%" height="40%" style="float:left">
</div>

<div align="center">
<img src="https://github.com/andrewunifei/CS229-Machine-Learning/blob/main/Regression/Linear/Non-parametric/Resources/Tau/c-tau_05.png" width="40%" height="40%" style="float:left">
<img src="https://github.com/andrewunifei/CS229-Machine-Learning/blob/main/Regression/Linear/Non-parametric/Resources/Tau/d-tau_005.png" width="40%" height="40%" style="float:left">
</div>
