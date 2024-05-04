from sklearn.tree import DecisionTreeClassifier
from models import getModels, loadData
from ALearn import Learner
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
def main():
    # performs 3 experiments for each algorithm
    X_train, y_train, X_test, y_test = loadData()  # loads training and test data split on 2008
    models = getModels() # loads models to list


def evaluate_predictions(y_true, y_pred):
    # returns both f1 micro score and hamming loss
    f1_micro = f1_score(y_true, y_pred, average='micro')
    hamming_loss_val = hamming_loss(y_true, y_pred)
    return f1_micro, hamming_loss_val

def pilotStudy():
    # performs pilot study to determine the level of uncertainty to determine convergence
    # should also determine how much data to use for active learning
    data = pd.read_csv("data/AlaskaClean_modified.csv")
    # Selecting features and labels
    features = ['water', 'wetland', 'shrub', 'dshrub', 'dec', 'mixed', 'spruce', 'baresnow']
    labels = ['AMPI', 'AMRO', 'ATSP', 'DEJU', 'FOSP', 'GCSP', 'HETH', 'OCWA', 'ROPT', 'SAVS', 'WCSP', 'WIPT', 'WISN', 'WIWA', 'YRWA']
    #| Park | 2004 | 2005 | 2006 | 2008 |
    #+------+-----+-----+-----+-----+
    #| ANIA | 0 | 0 | 0 | 100 |
    #| KATM | 0 | 313 | 69 | 0 |
    #| LACL | 318 | 0 | 35 | 0 |
    #+------+-----+-----+-----+-----+
    # Splitting the data into training and testing sets based on the year
    train_data = data[data['year'] == 2005]
    base_data = data[data['park'] == "ANIA"]
    test_data = data[data['park'] == "ANIA"]
    pool_data = data[data['park'] == "ANIA"]
    real_test = data[data['park'] == "ANIA"]
    # Extracting features and labels for training and testing
    X_train = np.array(train_data[features])
    y_train = np.array(train_data[labels])
    X_base = np.array(base_data[features])
    y_base = np.array(base_data[labels])
    X_test = np.array(test_data[features])
    y_test = np.array(test_data[labels])
    X_pool = np.array(pool_data[features])
    y_pool = np.array(pool_data[labels])

    def study():
        # uses test1 - uses 2006 data from katm to active sample
        # differnt park with 2 years difference from OG
        # trains on 2004 LACL active samples on 2005 KATM test on 2006 KATM

        initialmodels = getModels()
        [model.fit(X_train, y_train) for model in initialmodels]
        perfgoal = [evaluate_predictions(y_base, model.predict(X_base)) for model in initialmodels]
        print(perfgoal)
        learners = [Learner(model, X_train, y_train) for model in initialmodels]

        def activeLearning(learner, X_pool, y_pool, X_test, y_test, n_queries=False):
            model_f1, model_ham = evaluate_predictions(y_test, learner.model.predict(X_test))
            print(f"F1 initial: {model_f1:0.4f}, Ham initial: {model_ham:0.4f}")
            model_accuracy = (model_f1, model_ham)
            # Save our model's performance for plotting.
            performance = []
            performance.append(model_accuracy)


            if not n_queries:
                n_queries = X_pool.shape[0]
            X_pool_copy = X_pool.copy()
            y_pool_copy = y_pool.copy()
            for i in range(n_queries):

                query_idx = learner.query(X_pool_copy) # Correctly reshaping y

                learner.teach(X=X_pool_copy[query_idx].reshape(1, -1),y=y_pool_copy[query_idx].reshape(1, -1))

                # Remove the queried instance from the unlabeled pool
                X_pool = np.delete(X_pool_copy, query_idx, axis=0)
                y_pool = np.delete(y_pool_copy, query_idx, axis=0)
                # Calculate and report our model's accuracy.
                model_f1, model_ham = evaluate_predictions(y_test, learner.model.predict(X_test))
                print(f"F1 after query {i}: {model_f1:0.4f}, Ham after query {i}: {model_ham:0.4f}")
                model_accuracy = (model_f1, model_ham)
                # Save our model's performance for plotting.
                performance.append(model_accuracy)

            fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

            # Assuming performance_history contains tuples (f1_score, hamming_loss)
            f1_scores = [score[0] for score in performance]
            hamming_losses = [score[1] for score in performance]

            # Plot F1 Score
            ax.plot(f1_scores, label='F1 Score (Micro)', color='blue')
            ax.scatter(range(len(f1_scores)), f1_scores, s=13, color='blue')

            # Plot Hamming Loss
            ax.plot(hamming_losses, label='Hamming Loss', color='red')
            ax.scatter(range(len(hamming_losses)), hamming_losses, s=13, color='red')

            # Set x and y axis major locators and formatters
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

            # Set limits for y-axis
            ax.set_ylim(bottom=0, top=1)
            ax.grid(True)

            # Titles and labels
            ax.set_title('Incremental Classification Metrics')
            ax.set_xlabel('Query iteration')
            ax.set_ylabel('Metric value')

            # Adding a legend to differentiate the lines
            ax.legend()

            plt.show()

        for learner in learners:
            activeLearning(learner, X_pool, y_pool, X_test, y_test)

    study()

pilotStudy()