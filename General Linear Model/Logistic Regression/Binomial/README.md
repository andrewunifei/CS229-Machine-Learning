# Binary Classification
Based on notes from lecture 3 of the course [Stanford CS229: Machine Learning | Autumn 2018](https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) with professor Andrew Y. Ng, I utilized **scikit-learn** library to explore concepts of **logistic regression** as well as decision boundary in the context of binary classification.

### Dependencies 
[scikit-learn](https://github.com/scikit-learn/)</br>
[matplotlib](https://github.com/matplotlib/)</br>
[numpy](https://github.com/numpy)

## Logistic Regression
Logistic Regression in the context of binary classification is the supervised task of classify a set of datas into two categories (commomly defined as 1 and 0). It makes use of a **logistic function**, usually the **sigmoid function**, which takes as input a linear combination of features and parametes of the form:

<p align="center"><img src="https://user-images.githubusercontent.com/29299799/113064296-201b2c00-918d-11eb-93e4-013774bd466c.png"></p>

and returns a probability value which is ultimately evaluated into one of the two categories. The sigmoid function is defined as:

<p align="center"><img src="https://user-images.githubusercontent.com/29299799/113064556-991a8380-918d-11eb-8e54-493767ad1c25.png"></p>
 
In order to fit a set of parameters θ to the logistic regression model, we define a hypothesis function *h(x)* as:

<p align="center"><img src="https://user-images.githubusercontent.com/29299799/113065162-ca478380-918e-11eb-9acf-8546ec65ec9c.png"></p>

And the condition based on the returned value of the sigmoid function to determine a category is:

<p align="center"><img src="https://user-images.githubusercontent.com/29299799/113064970-6cb33700-918e-11eb-8d2b-68acad4272db.png"></p>

And through algebraic manipulations, make use of the method *maximum likelihood estimator*, which gives us the log likelihood defined as:

<p align="center"><img src="https://user-images.githubusercontent.com/29299799/113065538-725d4c80-918f-11eb-8ae6-1ee0588089f6.png"></p>

Which allow us to update θ through gradient ascent (since we want to maximaze a likelihood value). Updates in θ are given by Stochastic Gradient Ascent:

<p align="center"><img src="https://user-images.githubusercontent.com/29299799/113066968-db45c400-9191-11eb-844c-b68c607d3260.png"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/29299799/113065868-eef02b00-918f-11eb-8058-0c1658ea0151.png"></p>

### Decision Boundary
The Decision Boundary is a line which separates the categories of data. It is indepedent of the dataset and rely only on the parameters θ. In binary classification is formaly defined as:

<p align="center"><img src="https://user-images.githubusercontent.com/29299799/113066286-abe28780-9190-11eb-8aa6-789fbf242937.png"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/29299799/113066336-c1f04800-9190-11eb-84cf-b1abef2014ab.png"></p>

Because (Condition).

### Scikit-learn
#### Logistic Regression
```python
def logistic_regression(self) -> object:
  model = LogisticRegression(random_state=0)
  model.fit(self.X_train, self.y_train.ravel())
  predictions = model.predict(self.X_toPredict)
  self.coef = model.coef_[0]
  self.intercept = model.intercept_

  return predictions
```

Scikit-learn has a solver called SAG (Stochastic Average Gradient) which is a variant of the Stochastic Gradient method. SAG is a good solver when the dataset is large, given that the data classified in this case is small, the default solver was used (lbfgs).

#### Decision Boundary
```python
x = np.linspace(np.amin(x1), np.amax(x1))
y = -self.coef[0]/self.coef[1] * x - self.intercept/self.coef[1]
line, = ax.plot(x, y)
line.set_label('Decision Boundary')
ax.legend(loc='upper left')
```

## Visualization
<div align="center">
<img src="https://user-images.githubusercontent.com/29299799/110936147-31e87c80-830f-11eb-884f-b530804a2ff6.png" width="40%" height="40%" style="float:left">
<img src="https://user-images.githubusercontent.com/29299799/110936151-3319a980-830f-11eb-9467-08823427cf2e.png" width="40%" height="40%">
</div>

<div align="center">
<img src="https://user-images.githubusercontent.com/29299799/110936148-32811300-830f-11eb-9a99-8c270fc21a38.png" width="40%" height="40%" style="float:left">
<img src="https://user-images.githubusercontent.com/29299799/110936149-3319a980-830f-11eb-96e2-de4a28591c0c.png" width="40%" height="40%">
</div>
