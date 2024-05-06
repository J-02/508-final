import numpy as np
from scipy.stats import entropy
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, hamming_loss, roc_curve, auc
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess data
data_path = '/Users/tadeozuniga/PycharmProjects/508-final/data/AlaskaClean.csv'
data = pd.read_csv(data_path)
labels = ['AMPI', 'AMRO', 'ATSP', 'DEJU', 'FOSP', 'GCSP', 'HETH', 'OCWA', 'ROPT', 'SAVS', 'WCSP', 'WIPT', 'WISN', 'WIWA', 'YRWA']
features = ['water', 'wetland', 'shrub', 'dshrub', 'dec', 'mixed', 'spruce', 'baresnow', 'elev_m']

X = data[features].values
y = data[labels].to_numpy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Initialize OneVsRestClassifier with SVC, enabling probability estimates
classifier = OneVsRestClassifier(SVC(probability=True, max_iter=10000))
classifier.fit(X_train, y_train)

# Active learning parameters
num_iterations = 5
samples_per_iter = 20

# Query strategies
def query_samples(X_pool, classifier, n_samples=20):
    proba = classifier.predict_proba(X_pool)
    entropy_values = entropy(proba, axis=1)
    query_indices = np.argsort(entropy_values)[-n_samples:]
    return query_indices

# Active learning loop
for iteration in range(num_iterations):
    # Query new samples
    query_indices = query_samples(X_test, classifier, n_samples=samples_per_iter)
    X_query, y_query = X_test[query_indices], y_test[query_indices]

    # Add new data to training set
    X_train = np.vstack([X_train, X_query])
    y_train = np.vstack([y_train, y_query])

    # Retrain the classifier
    classifier.fit(X_train, y_train)

# Final evaluation and predictions
y_proba = classifier.predict_proba(X_test)
y_pred = classifier.predict(X_test)
uncertainties = entropy(y_proba, axis=1)

# Print final evaluation metrics
print(f"Hamming Loss: {hamming_loss(y_test, y_pred)}")
print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))

# Generate and save classification report
report = classification_report(y_test, y_pred, target_names=labels, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("ActiveLearning-SVM-classification_report.csv", index=True)

# Aggregated probabilities and uncertainties per species
aggregated_data = {
    "Species": labels,
    "Mean Probability": [],
    "Mean Uncertainty": []
}

for i, label in enumerate(labels):
    mean_prob = y_proba[:, i].mean()
    uncertainty_per_class = np.mean([uncertainties[idx] if y_test[idx][i] == 1 else 0 for idx in range(len(X_test))])
    aggregated_data["Mean Probability"].append(mean_prob)
    aggregated_data["Mean Uncertainty"].append(uncertainty_per_class)

aggregated_df = pd.DataFrame(aggregated_data)
aggregated_df.to_csv("ActiveLearning-SVM-aggregated_predictions_per_species.csv", index=False)

# Plot ROC curves for each class
fpr, tpr, roc_auc = {}, {}, {}
plt.figure(figsize=(12, 10))
for i in range(len(labels)):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f'{labels[i]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve per Class')
plt.legend(loc="lower right")
plt.show()

# Plot uncertainty for a specific sample
sample_index = 0  # Adjust this to view different test samples
plt.figure(figsize=(12, 8))
plt.errorbar(range(len(labels)), y_proba[sample_index], yerr=uncertainties[sample_index], fmt='o', color='b', ecolor='r', capthick=2)
plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
plt.title('Predictions with Uncertainties for Each Species')
plt.xlabel('Species')
plt.ylabel('Predicted Probability')
plt.show()


