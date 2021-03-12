# Binary Classification
Based on notes from lecture 3 of the course [Stanford CS229: Machine Learning | Autumn 2018](https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) with professor Andrew Y. Ng, I utilized **scikit-learn** library to explore concepts of **logistic regression** as well as decision boundary in the context of binary classification.

### Dependencies 
[scikit-learn](https://github.com/scikit-learn/)</br>
[matplotlib](https://github.com/matplotlib/)</br>
[numpy](https://github.com/numpy)

## Logistic Regression
Logistic Regression in the context of binary classification is the supervised task of classify a set of datas into two categories (commomly defined as 1 and 0). It makes use of a **logistic function**, usually the **sigmoid function**, which takes as input a linear combination of features and parametes of the form:

<p align="center"><img src="http://www.sciweavers.org/download/Tex2Img_1615550480.png"></p>

and returns a probability value which is ultimately evaluated into one of the two categoies. The sigmoid function is defined as:

<p align="center"><img src="http://www.sciweavers.org/download/Tex2Img_1615551113.png"></p>
  
And the criterion based on the returned value of the sigmoid function to determine a category is:

<p align="center"><img src="http://www.sciweavers.org/download/Tex2Img_1615551800.png"></p>
 
<div align="center">
<img src="https://user-images.githubusercontent.com/29299799/110936147-31e87c80-830f-11eb-884f-b530804a2ff6.png" width="40%" height="40%" style="float:left">
<img src="https://user-images.githubusercontent.com/29299799/110936151-3319a980-830f-11eb-9467-08823427cf2e.png" width="40%" height="40%">
</div>

<div align="center">
<img src="https://user-images.githubusercontent.com/29299799/110936148-32811300-830f-11eb-9a99-8c270fc21a38.png" width="40%" height="40%" style="float:left">
<img src="https://user-images.githubusercontent.com/29299799/110936149-3319a980-830f-11eb-96e2-de4a28591c0c.png" width="40%" height="40%">
</div>