    svm.fit(X_train, y_train)
    probabilities = svm.predict_proba(X_test)
    prob_positive = probabilities[:, 1]

    # Calculate the squared errors for different numbers of training samples
    squared_errors = []
    for i in range(1, len(X_train)):
        svm = SupportVectorMachine()
        svm.fit(X_train[:i], y_train[:i])
        probabilities = svm.predict_proba(X_test)
        if probabilities.shape[1] == 1:
            continue
        prob_positive = probabilities[:, 1]
        squared_error = np.mean((prob_positive - y_test) ** 2)
        squared_errors.append(squared_error)

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

    # Plot the squared errors
    plt.plot(squared_errors)
    plt.title('Squared error by number of training samples')
    plt.xlabel('Number of training samples')
    plt.ylabel('Squared error')
    plt.show()