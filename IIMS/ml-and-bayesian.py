# Import necessary libraries
import os
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# Set the working directory
os.chdir("C:/Users/jakub/Visual Studio Code/IIMS")

# Function to generate a list of random numbers
def generate_numbers(total):
    numbers = [random.randint(1, 100) for _ in range(total)]
    return numbers

# Function to train a decision tree classifier
def train_decision_tree(X_train, y_train):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf

# Function to train a Gaussian Naive Bayes classifier
def train_bayesian(X_train, y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    return gnb

# Define the total number of random numbers to generate and the number of points to plot
total = 100000
points = 50

# Generate the random numbers
numbers = generate_numbers(total)

# Prepare the training data for the classifiers
X_train = [[number] for number in numbers[:-10]]  # Use all but the last 10 numbers for training
y_train = [(numbers[i+1] > number) for i, number in enumerate(numbers[:-10])]  # The target is whether the next number is larger

# Train the decision tree classifier
clf = train_decision_tree(X_train, y_train)

# Train the Gaussian Naive Bayes classifier
gnb = train_bayesian(X_train, y_train)

# Initialize lists to store the actual outcomes and the predictions
actual_outcomes = []
ml_predictions = []
bayesian_predictions = []

# Generate predictions for the last 10 numbers
for i in range(-total + (total-points), 0):
    number = numbers[i]  # The current number
    
    # Predict whether the next number will be larger using the decision tree classifier
    ml_prediction = clf.predict([[number]])
    ml_predictions.append(ml_prediction[0])
    
    # Predict whether the next number will be larger using the Gaussian Naive Bayes classifier
    bayesian_prediction = gnb.predict([[number]])
    bayesian_predictions.append(bayesian_prediction[0])

    # Record the actual outcome
    actual_outcome = numbers[i+1] > number if i+1 < len(numbers) else 0
    actual_outcomes.append(actual_outcome)

    # Print the actual outcome, ML prediction, and Bayesian prediction for this step
    print(f"Step {i}: Actual Outcome: {actual_outcome}, ML Prediction: {ml_prediction[0]}, Bayesian Prediction: {bayesian_prediction[0]}")

# Plot the actual outcomes and the predictions
plt.figure(figsize=(12, 6))
plt.plot(actual_outcomes, 'o-', label='Actual Outcome')
plt.plot(ml_predictions, 'x-', label='ML Prediction')
plt.plot(bayesian_predictions, 's-', label='Bayesian Prediction')
plt.xlabel('Test Instance')
plt.ylabel('Outcome')
plt.legend()
plt.show()