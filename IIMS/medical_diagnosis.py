import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        # Initialize dictionaries to store class probabilities and feature probabilities
        self.class_probabilities = {}
        self.feature_probabilities = {}

    def fit(self, X, y):
        # Get the shape of the input data (number of samples and number of features)
        n_samples, n_features = X.shape
        # Get the unique classes present in the target labels
        classes = set(y)

        # Calculate class probabilities
        for c in classes:
            # Calculate the probability of each class by counting occurrences and dividing by the total number of samples
            self.class_probabilities[c] = sum(y == c) / n_samples

        # Calculate feature probabilities for each class
        for c in classes:
            # Get the subset of the input data for the current class
            X_c = X[y == c]
            # Initialize a dictionary to store feature probabilities for the current class
            self.feature_probabilities[c] = {}
            # Iterate over each feature
            for feature_index in range(n_features):
                # Extract feature values for the current class
                feature_values = X_c[:, feature_index]
                # Calculate the probabilities of unique feature values for the current class
                unique_values, counts = np.unique(feature_values, return_counts=True)
                probabilities = counts / len(feature_values)
                # Store the feature probabilities in a dictionary
                self.feature_probabilities[c][feature_index] = dict(zip(unique_values, probabilities))

    def predict(self, X):
        # Initialize a list to store predicted classes for each sample in the test data
        predictions = []
        # Iterate over each sample in the test data
        for x in X:
            # Initialize variables to keep track of the class with maximum probability and its probability
            max_class = None
            max_probability = -1
            # Iterate over each class and calculate the probability of the current sample belonging to that class
            for c, class_prob in self.class_probabilities.items():
                # Initialize probability with class probability
                probability = class_prob
                # Iterate over each feature in the sample
                for feature_index, feature_value in enumerate(x):
                    # If feature value not seen in training data, assume a small non-zero probability
                    feature_prob = self.feature_probabilities[c].get(feature_index, {}).get(feature_value, 1e-6)
                    # Multiply the probability of the current feature value given the class
                    probability *= feature_prob
                # Update the maximum probability and corresponding class if necessary
                if probability > max_probability:
                    max_probability = probability
                    max_class = c
            # Append the predicted class to the list of predictions
            predictions.append(max_class)
        # Return the list of predictions for all samples in the test data
        return predictions


# Sample dataset (replace with your actual dataset)
X = np.array([
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [0, 0, 1, 1],
    [1, 1, 1, 1]
])
y = np.array([1, 0, 1, 0, 1])

# Initialize and train the classifier
classifier = NaiveBayesClassifier()
classifier.fit(X, y)

# Test data
X_test = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1]
])

# Make predictions
predictions = classifier.predict(X_test)
print("Predictions:", predictions)
