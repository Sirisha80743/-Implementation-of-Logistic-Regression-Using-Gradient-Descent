# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the program.

2.Data preprocessing:

3.Cleanse data,handle missing values,encode categorical variables.

4.Model Training:Fit logistic regression model on preprocessed data.

5.Model Evaluation:Assess model performance using metrics like accuracyprecisioon,recall.

6.Prediction: Predict placement status for new student data using trained model.

7.End the program.

## Program:

Program to implement the the Logistic Regression Using Gradient Descent.

Developed by: SIRISHA P

RegisterNumber:  212224040321

```
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(np.dot(X, weights))
    epsilon = 1e-5
    cost = (-y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon)).mean()
    return cost

def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        h = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (h - y)) / m
        weights -= learning_rate * gradient
    return weights

def predict(X, weights):
    return sigmoid(np.dot(X, weights)) >= 0.5

np.random.seed(0)
num_samples = 100
X = np.random.randn(num_samples, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

X = np.hstack((np.ones((X.shape[0], 1)), X))

weights = np.zeros((X.shape[1], 1))

weights = gradient_descent(X, y, weights, learning_rate=0.1, iterations=1000)

predictions = predict(X, weights).astype(int)

accuracy = (predictions == y).mean()

print("Accuracy:", accuracy)
print("\nPredicted:")
print(predictions.T[0])

print("\nActual:")
print(y.T[0])

new_input = np.array([[1, 0.5, -1.2]])  # 1 is for bias term
new_prediction = predict(new_input, weights).astype(int)
print("\nPredicted Result:", new_prediction[0])
```
## Output:

![image](https://github.com/user-attachments/assets/7e35ed80-6fda-40e6-b9a8-ddf3245b67c2)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

