I explored the concept of multicass classification with more depth in the notebook called *development_analysis.ipynb*.<br>
If GitHub isn't displaying the *development_analysis.ipynb* file when you click on it, you can visualize it here: [Development Analysis](https://nbviewer.jupyter.org/github/andrewunifei/CS229-Machine-Learning/blob/main/General%20Linear%20Model/Logistic%20Regression/Multinomial/development_analysis.ipynb)

# Multiclass Classification
Multinomial Classification is the supervised task of categorize a dataset into multiple categories. Its activation function is called **Softmax** and is defined as:

<p align="center"><img src="https://user-images.githubusercontent.com/29299799/113641493-13e31300-9654-11eb-9334-c968bc267d95.png"></p>

Softmax function is a generalization of the **logistic function** (used in binary classification) for higher dimensions.

Following the third design definition of a General Linear Model described [here](https://github.com/andrewunifei/CS229-Machine-Learning/tree/main/General%20Linear%20Model), the natural parameter η have a linear relation with the set of features x, therefore can be expressed as η = θ^Tx. From this we arrive at:

<p align="center"><img src="https://user-images.githubusercontent.com/29299799/113644337-eb124c00-965a-11eb-8caa-e406c06a85a0.png"></p>

In order to learn the set of parameters θ, we can maximize the log-likelihood estimate of the parameters using a method of gradient ascent.
