
import os
os.chdir("C:/Users/jakub/Visual Studio Code/IIMS")      # Change the working directory

import numpy as np                                      # Import numpy for numerical operations
from sklearn.model_selection import train_test_split    # Import function to split data into training and testing sets
import matplotlib.pyplot as plt                         # Import matplotlib for plotting
from sklearn import datasets                            # Import sklearn's datasets module
import pandas as pd                                     # Import pandas for data manipulation
from sklearn.preprocessing import LabelEncoder          # Import LabelEncoder for encoding categorical variables

#https://www.kaggle.com/code/ismailsefa/heart-disease-predic-machine-learning-naive-bayes
#https://github.com/MastersAbh/Heart-Disease-Prediction-using-Naive-Bayes-Classifier

class NaiveBayes:                                   # Define a class for the Naive Bayes model
    def fit(self, X, y):                            # Define the fit method to train the model
        n_samples, n_features = X.shape             # Get the number of samples and features
        self._classes = np.unique(y)                # Get the unique classes
        n_classes = len(self._classes)              # Get the number of classes

        # Initialize mean, variance, and priors for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):                 # For each class
            X_c = X[y == c]                                     # Get samples of this class
            self._mean[idx, :] = X_c.mean(axis=0)               # Calculate the mean
            self._var[idx, :] = X_c.var(axis=0)                 # Calculate the variance
            self._priors[idx] = X_c.shape[0] / float(n_samples) # Calculate the prior

    def predict_proba(self, X):                                 # Define method to predict probabilities
        return np.array([self._predict_proba(x) for x in X])    # Return an array of probabilities for each sample

    def _predict_proba(self, x):                                # Define helper method to calculate probabilities for a single sample
        posteriors = []                                         # Initialize list to store posterior probabilities

        # Calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])                       # Calculate the log prior
            class_conditional = np.sum(np.log(self._pdf(idx, x)))   # Calculate the log likelihood
            posterior = prior + class_conditional                   # Calculate the log posterior
            posteriors.append(posterior)                            # Append the posterior to the list

        posteriors = np.exp(posteriors)                             # Convert log probabilities to probabilities
        return posteriors / sum(posteriors)                         # Normalize the probabilities

    def _pdf(self, class_idx, x):                           # Define helper method to calculate the probability density function
        mean = self._mean[class_idx]                        # Get the mean for this class
        var = self._var[class_idx]                          # Get the variance for this class
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))  # Calculate the numerator of the PDF
        denominator = np.sqrt(2 * np.pi * var)              # Calculate the denominator of the PDF
        return numerator / denominator                      # Return the PDF

if __name__ == "__main__":                                  # If this script is run directly (not imported)

    def accuracy(y_true, y_pred):                           # Define a function to calculate accuracy
        accuracy = np.sum(y_true == y_pred) / len(y_true)   # Calculate accuracy
        return accuracy

    df = pd.read_csv('heartdisease.csv')                    # Load the dataset

    le = LabelEncoder()                                     # Initialize a LabelEncoder
    df['num'] = le.fit_transform(df['num'])                 # Encode the target variable

    # Calculate the absolute correlation with the target
    correlations = df.drop('num', axis=1).apply(lambda x: x.corr(df['num'])).abs()

    threshold = 0.1                                         # Set a threshold for correlation

    # Identify features to drop
    to_drop = correlations[correlations < threshold].index.tolist()

    # Drop features
    X = df.drop(['num'] + to_drop, axis=1).values
    y = df['num'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    nb = NaiveBayes()                                       # Initialize a NaiveBayes object
    nb.fit(X_train, y_train)                                # Fit the model to the training data
    probabilities = nb.predict_proba(X_test)                # Predict probabilities for the test data

    # Extract probabilities of the positive class
    prob_positive = probabilities[:, 1]

    # Plot histogram of predicted probabilities
    plt.hist(prob_positive, bins=10, edgecolor='k')
    plt.title('Histogram of predicted probabilities')
    plt.xlabel('Predicted probability of heart disease')
    plt.ylabel('Frequency')
    plt.show()                                              # Show the plot