import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report, hamming_loss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Normalization, Activation, BatchNormalization, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# Custom Dropout class for Monte Carlo Dropout
class MCDropout(Layer):
    def __init__(self, rate):
        super(MCDropout, self).__init__()
        self.rate = rate

    def call(self, inputs, training=None):
        return tf.nn.dropout(inputs, rate=self.rate, name="MCDropout")

# Load data
data_path = '/Users/tadeozuniga/PycharmProjects/508-final/data/AlaskaClean.csv'
data = pd.read_csv(data_path)

# Define your labels
labels = ['AMPI', 'AMRO', 'ATSP', 'DEJU', 'FOSP', 'GCSP', 'HETH', 'OCWA', 'ROPT', 'SAVS', 'WCSP', 'WIPT', 'WISN', 'WIWA', 'YRWA']
features = ['water', 'wetland', 'shrub', 'dshrub', 'dec', 'mixed', 'spruce', 'baresnow', 'elev_m']

# Preprocess data
X = data[features].values
y = data[labels].to_numpy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Normalize features
normalizer = Normalization()
normalizer.adapt(X_train)

# Build the model with regularization and batch normalization
def build_another_model():
    model = Sequential([
        Input(shape=(X_poly.shape[1],)),
        normalizer,
        Dense(512, activation=None, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Activation('relu'),
        MCDropout(0.5),
        Dense(512, activation=None, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Activation('relu'),
        MCDropout(0.5),
        Dense(len(labels), activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', AUC(name='auc')])
    return model

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model = build_another_model()
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping])

# Evaluate model and predict
test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test AUC: {test_auc}")
num_samples = 30
predictions = np.array([model.predict(X_test) for _ in range(num_samples)])
mean_predictions = predictions.mean(axis=0)
std_predictions = predictions.std(axis=0)
hamming_loss_value = hamming_loss(y_test, np.round(mean_predictions))
print(f"Hamming Loss: {hamming_loss_value}")

# Generate classification report and extract F1-scores
predicted_classes = np.round(mean_predictions)
report = classification_report(y_test, predicted_classes, target_names=labels, output_dict=True, zero_division=0)
print(classification_report(y_test, predicted_classes, target_names=labels, zero_division=0))

# Extract F1-scores for plotting
f1_scores = [report[label]['f1-score'] for label in labels]

# Plot ROC curves for each class
fpr, tpr, roc_auc = {}, {}, {}

plt.figure(figsize=(12, 10))  # Adjust size to better fit multiple curves
for i in range(len(labels)):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], mean_predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f'{labels[i]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve per Class')
plt.legend(loc="lower right")
plt.show()  # Display the ROC curve plot

# Plot Uncertainty for a Specific Sample
plt.figure(figsize=(12, 8))
sample_index = 0  # Adjust as necessary to select a different test sample
errors = std_predictions[sample_index]
plt.errorbar(range(len(labels)), mean_predictions[sample_index], yerr=errors, fmt='o', color='b', ecolor='r', capthick=2)
plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
plt.title('Predictions with Uncertainties for Each Species')
plt.xlabel('Species')
plt.ylabel('Predicted Probability')
plt.show()  # Display the uncertainty plot

# Plot F1-Scores for Each Species
plt.figure(figsize=(12, 8))  # Set the figure size
plt.bar(labels, f1_scores, color='skyblue')  # Create a bar chart
plt.xlabel('Species')  # Label on X-axis
plt.ylabel('F1-Score')  # Label on Y-axis
plt.title('F1-Scores for Each Species')  # Title of the plot
plt.xticks(rotation=45)  # Rotate labels on X-axis for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Adding a grid for y-axis
plt.show()  # Display the F1-score plot

# Print probabilities and uncertainties for a subset of samples
num_samples_to_print = 5  # Adjust this value as needed
for idx in range(num_samples_to_print):
    print(f"Sample {idx + 1}:")
    for label, mean_prob, std_dev in zip(labels, mean_predictions[idx], std_predictions[idx]):
        print(f"  {label}: Probability = {mean_prob:.3f}, Uncertainty (Std Dev) = {std_dev:.3f}")
    print("\n")

###############

###Print Output

import pandas as pd

# Save classification report to a DataFrame and export as CSV
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("MCDropout-DeepNN-classification_report.csv", index=True)

# Prepare to store prediction data in a DataFrame
predictions_data = {
    "Sample": [],
    "Species": [],
    "Probability": [],
    "Uncertainty": []
}

# Aggregated per species across all samples
aggregated_data = {
    "Species": labels,
    "Mean Probability": [],
    "Mean Uncertainty": []
}

for i, label in enumerate(labels):
    mean_prob = mean_predictions[:, i].mean()
    mean_uncertainty = std_predictions[:, i].mean()
    aggregated_data["Mean Probability"].append(mean_prob)
    aggregated_data["Mean Uncertainty"].append(mean_uncertainty)

# Save the aggregated data to a CSV file
aggregated_df = pd.DataFrame(aggregated_data)
aggregated_df.to_csv("MCDropout-DeepNN-aggregated_predictions_per_species.csv", index=False)

