import numpy as np
from sklearn.utils.extmath import softmax
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from xgboost import Booster
from xgboost import XGBClassifier
class Learner:

    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.model.fit(X_train, y_train)
        self.sample_weight = np.ones(len(self.y_train)) if self.y_train is not None else None


    def teach(self, X, y, sample_weight):

    # Update the training data
        self.X_train = np.vstack([self.X_train, X])
        self.y_train = np.vstack([self.y_train, y])

    # Update the weights for the existing and new data
        if self.sample_weight is not None:
            self.sample_weight = np.append(self.sample_weight, sample_weight)
        else:
            self.sample_weight = np.full(len(self.y_train), 1)  # This case is just in case, should not actually happen

        # Fit the model on the updated training data with updated weights
        self.model.fit(self.X_train, self.y_train, sample_weight=self.sample_weight)

    def query(self, X_pool, is_first_query=False, weightx=3,n=1):
        """
        Applies a custom query strategy to determine which instance from the pool
        should be labeled next. This method selects the instance with the minimum
        confidence across all labels.

        Returns the indices of the instances to be labeled.
        """
        try:
            # Get the probability predictions for each class for each instance
            probas = self.model.predict_proba(X_pool)
            if type(probas) is list:
                entropy = np.sum([-np.sum(p * np.log(p + 1e-9), axis=1) for p in probas], axis=0)
                print(f"Entropy range: {np.min(entropy)}-{np.max(entropy)}")
                uncertainties = np.array([1 - np.max(proba, axis=1) for proba in probas])
            else:
                uncertainties = probas
                probas = np.clip(probas, 1e-9, 1 - 1e-9)
                # Calculate entropy for each label
                entropy = -probas * np.log(probas) - (1 - probas) * np.log(1 - probas)
                # Sum entropy across all labels to get total entropy for each instance
                entropy = np.sum(entropy, axis=1)
                print(f"Entropy range: {np.min(entropy)}-{np.max(entropy)}")
        except:
            # If the classifier does not support predict_proba, we use decision_function
            # Softmax is applied to convert decision scores to probabilities
            decision_function = self.model.decision_function(X_pool)
            probas = softmax(decision_function, axis=2)
            uncertainties = 1 - np.max(probas, axis=2)

        # Calculate cumulative uncertainty across all labels for each sample
        cumulative_uncertainty = np.sum(uncertainties, axis=0)
        # Find indices with the highest entropy
        if is_first_query:
            query_indices = np.argsort(-entropy)[:5]  # Top 5 uncertain indices
        else:
            query_indices = np.argsort(-entropy)[:1]  # Top 1 uncertain index

        return query_indices, entropy[query_indices]+1

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