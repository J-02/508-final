import pandas as pd
import numpy as np
from keras.src.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense
from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.metrics import f1_score, hamming_loss
from scipy.stats import entropy
from matplotlib import pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def calculate_aggregated_entropy(probabilities):
    """Calculate aggregated entropy across all class probabilities for each instance."""
    return np.sum([entropy(prob.T) for prob in probabilities], axis=0)

def select_uncertain_samples(model, X, n_samples):
    """Select samples based on highest uncertainty using the model's predict_proba method."""
    # Check if model can predict probabilities
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        uncertainties = calculate_aggregated_entropy(probabilities)
        # Get indices of samples with the highest uncertainty
        indices = np.argsort(uncertainties)[-n_samples:]
    else:
        # Fallback to random selection if predict_proba is not available
        indices = np.random.choice(len(X), size=n_samples, replace=False)
    return indices

def loadData(file):
    data = pd.read_csv(file)
    # Selecting features and labels
    features = ['water', 'wetland', 'shrub', 'dshrub', 'dec', 'mixed', 'spruce', 'baresnow', 'elev_m']
    labels = ['AMPI', 'AMRO', 'ATSP', 'DEJU', 'FOSP', 'GCSP', 'HETH', 'OCWA', 'ROPT', 'SAVS', 'WCSP', 'WIPT', 'WISN', 'WIWA', 'YRWA']

    # Splitting the data into training and testing sets based on the year
    train_data = data[data['year'] < 2008]
    test_data = data[data['year'] == 2008]

    # Extracting features and labels for training and testing
    X_train = np.array(train_data[features])
    y_train = np.array(train_data[labels])
    X_test = np.array(test_data[features])
    y_test = np.array(test_data[labels])

    # Show the first few rows of training features and labels to verify
    return  X_train, y_train, X_test, y_test

def fine_tune_model(model, X_train, y_train, X_fine_tune, y_fine_tune, X_final_test, y_final_test):
    """Fine-tune model and evaluate on the remaining test set."""
    # Fine-tune model on uncertain test samples
    model.fit(X_fine_tune, y_fine_tune)
    # Evaluate model on the remaining test data
    y_pred = model.predict(X_final_test)
    if hasattr(model, 'predict_proba'):  # For models that predict probabilities
        y_pred = (y_pred > 0.5).astype(int)
    f1_micro = f1_score(y_final_test, y_pred, average='micro')
    hamming = hamming_loss(y_final_test, y_pred)
    print(f'{model.__class__.__name__} Fine-tuned - F1 Micro: {f1_micro}, Hamming Loss: {hamming}')
    return f1_micro, hamming
results = []
def experiment(models, X_train, y_train, X_test, y_test, fine_tune_ratio):
    """Run experiment to fine-tune models on uncertain test data and compare performance."""
    n_samples = int(len(X_test) * fine_tune_ratio)
    for model in models:
        # Train on the original training data
        model.fit(X_train, y_train)

        # Select uncertain samples based on entropy
        uncertain_indices = select_uncertain_samples(model, X_test, n_samples)
        X_fine_tune, y_fine_tune = X_test[uncertain_indices], y_test[uncertain_indices]

        # Remaining indices for final evaluation
        remaining_indices = np.array([i for i in range(len(X_test)) if i not in uncertain_indices])
        X_final_test, y_final_test = X_test[remaining_indices], y_test[remaining_indices]

        # Evaluate model on the remaining test data before fine-tuning
        y_pred = model.predict(X_final_test)
        f1_micro = f1_score(y_final_test, y_pred, average='micro')
        hamming = hamming_loss(y_final_test, y_pred)
        print(f'{model.__class__.__name__} Initial - F1 Micro: {f1_micro}, Hamming Loss: {hamming}')

        # Fine-tune using uncertain terms from the test set
        f1_micro, hamming = fine_tune_model(model, X_train, y_train, X_fine_tune, y_fine_tune, X_final_test, y_final_test)
        results.append([model, fine_tune_ratio,f1_micro, hamming])


# Define models
rf_model = RandomForestClassifier(n_estimators=100)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
nn_model = Sequential([
    Dense(64, activation='relu', input_dim=9),
    Dense(32, activation='relu'),
    Dense(15, activation='sigmoid')  # Assuming 15 output classes
])

nn_model.compile(
    optimizer=Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Load data
X_train, y_train, X_test, y_test = loadData("data/AlaskaClean.csv")

# Run experiment with varying amounts of test data used for fine-tuning
for ratio in [0.05, 0.1, .15, 0.2, .25, .3, .35, .4, .45, 0.5]:
    print(f"\nExperiment with {ratio*100}% of test data for fine-tuning:")
    experiment([rf_model, xgb_model], X_train, y_train, X_test, y_test, fine_tune_ratio=ratio)

def run_model(model, X_train, y_train, X_test, y_test):
    # Check if the model is a neural network
    if isinstance(model, Sequential):
        # Convert labels to float for compatibility with keras
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
        # Compile the neural network
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        # Predict on testing data
        y_pred = model.predict(X_test) > 0.5  # Threshold to convert probabilities to binary output
        # Calculate metrics
        f1_micro = f1_score(y_test, y_pred, average='micro')
        hamming = hamming_loss(y_test, y_pred)
        print(f'Neural Network - F1 Micro: {f1_micro}, Hamming Loss: {hamming}')
    else:
        # Train the model
        model.fit(X_train, y_train)
        # Predict on testing data
        y_pred = model.predict(X_test)
        # Calculate metrics
        f1_micro = f1_score(y_test, y_pred, average='micro')
        hamming = hamming_loss(y_test, y_pred)
        print(f'{model.__class__.__name__} - F1 Micro: {f1_micro}, Hamming Loss: {hamming}')

    return model







