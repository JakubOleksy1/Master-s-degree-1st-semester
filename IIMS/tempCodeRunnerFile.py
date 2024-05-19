for i in range(errVar, len(X_train)):  # Start from 50 to ensure at least two classes
        rf = RandomForestClassifier()
        rf.fit(X_train[:i], y_train[:i])
        probabilities = rf.predict_proba(X_test)

        # Skip this iteration if the model predicts only one class
        if probabilities.shape[1] == 1:
            continue

        prob_positive = probabilities[:, 1]
        squared_error = np.mean((prob_positive - y_test/100) ** 2)
        squared_errors_rf.append(squared_error)

    # Plot the histograms and the cumulative distribution
    plt.subplot(1,3,1)
    plt.hist(prob_positive, bins=10, edgecolor='k', density=True)
    plt.title('Histogram of predicted probabilities')
    plt.xlabel('Predicted probability of heart disease')
    plt.ylabel('Frequency')

    plt.subplot(1,3,2)
    plt.hist(prob_positive, bins=10, edgecolor='k', cumulative=True, density=True)
    plt.title('Cumulative distribution of predicted probabilities')
    plt.xlabel('Predicted probability of heart disease')
    plt.ylabel('Cumulative frequency')

    plt.subplot(1,3,3)
    plt.plot(np.sort(prob_positive))
    plt.title('Probability by iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Predicted probability of heart disease')

    plt.tight_layout()
    plt.show()

    plt.plot(squared_errors_nb)

    # Plot the squared errors
    plt.plot(squared_errors_svm)

    plt.plot(squared_errors_rf)
    plt.title('Squared error by number of training samples')
    plt.xlabel('Number of training samples')
    plt.ylabel('Squared error')
    plt.plot(squared_errors_nb, label='Naive Bayes')
    plt.plot(squared_errors_svm, label='SVM')
    plt.plot(squared_errors_rf, label='Random Forest')
    plt.legend()  
    plt.show()