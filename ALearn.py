import numpy as np
from sklearn.utils.extmath import softmax
class Learner:

    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
    def teach(self, X, y):
        """
        Fits the model on the provided training data plus the points to teach
        """
        self.X_train = np.vstack([self.X_train, X])
        self.y_train = np.vstack([self.y_train, y])

        # Fit the model on the updated training data
        self.model.fit(self.X_train, self.y_train)

    def query(self, X_pool):
        """
        Applies a custom query strategy to determine which instance from the pool
        should be labeled next. This method selects the instance with the minimum
        confidence across all labels.

        Returns the indices of the instances to be labeled.
        """
        try:
            # Get the probability predictions for each class for each instance
            probas = self.model.predict_proba(X_pool)
            uncertainties = np.array([1 - np.max(proba, axis=1) for proba in probas])
        except:
            # If the classifier does not support predict_proba, we use decision_function
            # Softmax is applied to convert decision scores to probabilities
            decision_function = model.decision_function(X_pool)
            probas = softmax(decision_function, axis=2)
            uncertainties = 1 - np.max(probas, axis=2)

        # Calculate cumulative uncertainty across all labels for each sample
        cumulative_uncertainty = np.sum(uncertainties, axis=0)

        # Select the index of the instance with the maximum cumulative uncertainty
        query_idx = np.argmax(cumulative_uncertainty)

        return query_idx

    def run(self, model, X_train, y_train, X_pool, y_pool, n_queries=100):
        """
        Executes the active learning loop, querying the most uncertain sample,
        teaching the model, and updating the pool.
        """
        self.teach(model, X_train, y_train)

        for i in range(n_queries):
            query_idx = self.query(model, X_pool)
            self.teach(model, X_pool[query_idx].reshape(1, -1), y_pool[query_idx].reshape(1, -1))
            # Remove the queried instance from the pool
            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx, axis=0)
            print(f"Query {i + 1}: Instance {query_idx} has been added to the training set.")

# Usage
# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier()
# learner_instance = Learner()
# learner_instance.run(model, X_train, y_train, X_pool, y_pool)