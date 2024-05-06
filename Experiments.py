from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from models import getModels, loadData
from ALearn import Learner
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from xgboost import Booster
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import IterativeStratification
def evaluate(model, X_pool, y_pool, iteration="initial"):
    if isinstance(model, Booster):
    # For XGBoost, use the `fit` method with `xgb_model` parameter
        dnew = xgb.DMatrix(X_pool, label=y_pool)
        y_pred = model.predict(dnew)
        y_pred = y_pred.round(decimals=0).astype(int)
    else:
        y_pred = model.predict(X_pool)
    f1, ham = evaluate_predictions(y_pool, y_pred)
    print(f"Iteration {iteration}: F1 {f1:.4f}, Ham {ham:.4f}")
    return f1, ham

def evaluate_predictions(y_true, y_pred):
    # returns both f1 micro score and hamming loss
    f1_micro = f1_score(y_true, y_pred, average='micro')
    hamming_loss_val = hamming_loss(y_true, y_pred)
    return f1_micro, hamming_loss_val

def main():
    # performs pilot study to determine the level of uncertainty to determine convergence
    # should also determine how much data to use for active learning
    data = pd.read_csv("data/AlaskaClean_modified.csv")
    # Selecting features and labels
    features = ['water', 'wetland', 'shrub', 'dshrub', 'dec', 'mixed', 'spruce', 'baresnow']
    labels = ['AMPI', 'AMRO', 'ATSP',  'FOSP', 'GCSP', 'HETH', 'OCWA', 'ROPT', 'SAVS', 'WCSP', 'WIPT', 'WISN', 'WIWA']
    #labels = ['AMPI', 'AMRO', 'ATSP', 'DEJU', 'FOSP', 'GCSP', 'HETH', 'OCWA', 'ROPT', 'SAVS', 'WCSP', 'WIPT', 'WISN', 'WIWA', 'YRWA']
    #| Park | 2004 | 2005 | 2006 | 2008 |
    #+------+-----+-----+-----+-----+
    #| ANIA | 0 | 0 | 0 | 100 |
    #| KATM | 0 | 313 | 69 | 0 |
    #| LACL | 318 | 0 | 35 | 0 |
    #+------+-----+-----+-----+-----+
    # Define which data to use for train, pool, and test


    # Randomly select n samples from data_2005 for training
    #n = 318
    #np.random.seed(16)
    #train_indices = np.random.choice(data_2005.index, size=n, replace=False)
    #train_data = data_2005.loc[train_indices]
    train_data = data[data['year'] == 2006]
    test_data = data[data['year'] == 2008]
    stratifier = IterativeStratification(n_splits=4, order=1, random_state=2)
    #pool_data, test_data = train_test_split(df, test_size=0.50, random_state=2)
    # Use the remaining samples as pool data
    #pool_data = data[data['year'] == 2005]
    #pool_data = data_2005.drop(train_indices)
      # This unpacks the first split pair (modify if different splits are needed)
    X_test = test_data[features].values
    y_test = test_data[labels].values

    indices = list(stratifier.split(X_test, y_test))
    test_indices, train_indices = indices[0]
    # Using one split as the pool and the other as the test set
    X_pool, y_pool = X_test[train_indices], y_test[train_indices]
    X_test, y_test = X_test[test_indices], y_test[test_indices]
    # Define test data
    #test_data = data[(data['year'] == 2006) & (data['park'] == "KATM")]
    # Extracting features and labels for training and testing
    X_train = np.array(train_data[features])
    y_train = np.array(train_data[labels])

    #X_pool = np.array(pool_data[features])
    #y_pool = np.array(pool_data[labels])

    #X_test = np.array(test_data[features])
    #y_test = np.array(test_data[labels])

    initialmodels = getModels()
    [model.fit(X_train, y_train) for model in initialmodels]
    learners = [Learner(model, X_train, y_train) for model in initialmodels]
    model_names = ["Random Forest", "XGBoost", "LightGBM"]
    palette = sns.color_palette("husl", len(learners))

    for i, (learner, name) in enumerate(zip(learners, model_names)):
        f1_scores, hamming_losses = activeLearning(learner, X_pool, y_pool, X_test, y_test)
        learner.model.fit(X_pool, y_pool)
        f1_small, ham_small = evaluate(learner.model, X_test, y_test, "Small model")
        X_combined = np.vstack([X_pool, X_train])
        y_combined = np.vstack([y_pool, y_train])
        learner.model.fit(X_combined, y_combined)

        # Evaluate the model trained on the combined dataset
        f1_combined, ham_combined = evaluate(learner.model, X_test, y_test, "Combined model")
        # Create a DataFrame for plotting
        data = pd.DataFrame({'Iteration': range(len(f1_scores)),
                             'F1 Score': f1_scores,
                             'Hamming Loss': hamming_losses})

        # Melt the DataFrame for plotting with Seaborn
        melted_data = pd.melt(data, id_vars=['Iteration'], value_vars=['F1 Score', 'Hamming Loss'], var_name='Metric',
                              value_name='Score')
        window_size = 1  # Adjust the window size as needed
        melted_data['Smoothed Score'] = melted_data.groupby('Metric')['Score'].rolling(window=window_size,
                                                                                       center=True).mean().reset_index(
            level=0, drop=True)

        # Smooth the lines using a rolling average
        sns.lineplot(data=melted_data[melted_data['Metric'] == 'F1 Score'], x='Iteration', y='Smoothed Score',
                     label=f'{name} F1 Score', linestyle='-', linewidth=2, color=palette[i])

        # Optional: Plotting Hamming Loss if needed
        # sns.lineplot(data=melted_data[melted_data['Metric'] == 'Hamming Loss'], x='Iteration', y='Smoothed Score', label=f'{name} Hamming Loss', linestyle=':', linewidth=2, color=palette[i])
        plt.axhline(y=f1_combined, color=palette[i], linestyle='--', label=f'{name} Combined Model F1', linewidth=1)
        # Static model's performance lines
        plt.axhline(y=f1_small, color=palette[i], linestyle=':', label=f'{name} Small Model F1', linewidth=1)
        # plt.axhline(y=ham, color=palette[i], linestyle=':', label=f'{name} Small Model Hamming Loss', linewidth=1)
    plt.legend(title='Model', loc='upper left', bbox_to_anchor=(1.05, 1), title_fontsize='8', fontsize='6')
    plt.title('Active Learning Performance')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main(weightx=1):
    data = pd.read_csv("data/AlaskaClean_modified.csv")
    features = ['water', 'wetland', 'shrub', 'dshrub', 'dec', 'mixed', 'spruce', 'baresnow']
    labels = ['AMPI', 'AMRO', 'ATSP', 'FOSP', 'GCSP', 'HETH', 'OCWA', 'ROPT', 'SAVS', 'WCSP', 'WIPT', 'WISN', 'WIWA']

    train_data = data[(data["year"] == 2006) & (data["park"] == "KATM")]
    test_data = data[data['year'] == 2008]

    X_train = np.array(train_data[features])
    y_train = np.array(train_data[labels])
    X_test_all = np.array(test_data[features])
    y_test_all = np.array(test_data[labels])

    stratifier = IterativeStratification(n_splits=4, order=1)
    indices = list(stratifier.split(X_test_all, y_test_all))

    initialmodels = getModels()
    [model.fit(X_train, y_train) for model in initialmodels]
    model_names = ["Random Forest", "XGBoost", "LightGBM"]
    palette = sns.color_palette("husl", len(initialmodels))

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))  # Adjust as needed for the number of splits
    axs = axs.flatten()


    performances = {name: {"active": [], "combined": [], "small": []} for name in model_names}

    for split_index, (test_indices, train_indices) in enumerate(indices):
        X_pool, y_pool = X_test_all[train_indices], y_test_all[train_indices]
        X_test, y_test = X_test_all[test_indices], y_test_all[test_indices]

        for i, model in enumerate(initialmodels):
            model.fit(X_pool, y_pool)
            f1_small, ham_small = evaluate(model, X_test, y_test, f"Fold {split_index + 1}: Small {model_names[i]}")
            performances[model_names[i]]["small"].append(f1_small)

            X_combined = np.vstack([X_pool, X_train])
            y_combined = np.vstack([y_pool, y_train])
            model.fit(X_combined, y_combined)
            f1_combined, ham_combined = evaluate(model, X_test, y_test,
                                                 f"Fold {split_index + 1}: Combined {model_names[i]}")
            performances[model_names[i]]["combined"].append(f1_combined)

            learner = Learner(model, X_train, y_train)
            f1_scores, _ = activeLearning(learner, X_pool, y_pool, X_test, y_test)
            performances[model_names[i]]["active"].append(max(f1_scores))
            f1_scores, hamming_losses = activeLearning(learner, X_pool, y_pool, X_test, y_test, weightx=weightx)
            data = pd.DataFrame({'Iteration': range(len(f1_scores)),
                                 'F1 Score': f1_scores,
                                 'Hamming Loss': hamming_losses})
            melted_data = pd.melt(data, id_vars=['Iteration'], value_vars=['F1 Score', 'Hamming Loss'],
                                  var_name='Metric',
                                  value_name='Score')
            window_size = 3  # Smooth the lines
            melted_data['Smoothed Score'] = melted_data.groupby('Metric')['Score'].rolling(window=window_size,
                                                                                           center=True).mean().reset_index(
                level=0, drop=True)

            sns.lineplot(ax=axs[split_index], data=melted_data[melted_data['Metric'] == 'F1 Score'], x='Iteration',
                         y='Smoothed Score',
                         label=f'{model_names[i]} F1 Score', linestyle='-', linewidth=2, color=palette[i])
            axs[split_index].axhline(y=f1_combined, color=palette[i], linestyle='--', label=f'{model_names[i]} Combined Model F1',
                                     linewidth=1)
            axs[split_index].axhline(y=f1_small, color=palette[i], linestyle=':', label=f'{model_names[i]} Small Model F1',
                                     linewidth=1)
            axs[split_index].set_title(f'Fold {split_index + 1}, {weightx}')
            axs[split_index].set_xlabel('Iteration')
            axs[split_index].set_ylabel('Score')
            axs[split_index].grid(True)

    # Single legend for all plots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 1), title='Model and Fold', title_fontsize='10',
               fontsize='8')
    plt.tight_layout()
    plt.show()

    avg_performances = {name: {metric: np.mean(vals) for metric, vals in metrics.items()} for name, metrics in
                        performances.items()}

    # Print results in LaTeX table format
    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("Model & Avg Best Active F1 & Avg Combined F1 & Avg Small F1 \\\\")
    print("\\hline")
    for model, metrics in avg_performances.items():
        print(f"{model} & {metrics['active']:.4f} & {metrics['combined']:.4f} & {metrics['small']:.4f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
def activeLearning(learner, X_pool, y_pool, X_test, y_test, n_queries=None, weightx=1):
    if not n_queries:
        n_queries = X_pool.shape[0]

    X_pool_copy = X_pool.copy()
    y_pool_copy = y_pool.copy()

    f1_scores = []
    hamming_losses = []

    f1, ham = evaluate(learner.model, X_test, y_test)
    f1_scores.append(f1)
    hamming_losses.append(ham)

    query_count = 0

    while query_count < n_queries:
        is_first_query = (query_count == 0)
        query_indices, query_weights = learner.query(X_pool_copy, is_first_query=is_first_query, weightx=weightx)

        for idx, weight in zip(query_indices, query_weights):
            if query_count >= n_queries:
                break
            learner.teach(X=X_pool_copy[idx].reshape(1, -1), y=y_pool_copy[idx].reshape(1, -1), sample_weight=weight)
            query_count += 1

        X_pool_copy = np.delete(X_pool_copy, query_indices, axis=0)
        y_pool_copy = np.delete(y_pool_copy, query_indices, axis=0)

        f1, ham = evaluate(learner.model, X_test, y_test, query_count)
        f1_scores.append(f1)
        hamming_losses.append(ham)

    return f1_scores, hamming_losses

main()
