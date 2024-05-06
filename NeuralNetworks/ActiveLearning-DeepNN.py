import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import roc_curve, auc, classification_report, hamming_loss
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Normalization, Activation, BatchNormalization, Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# Define the custom Monte Carlo Dropout layer
class MCDropout(Layer):
    def __init__(self, rate):
        super(MCDropout, self).__init__()
        self.rate = rate

    def call(self, inputs, training=None):
        return tf.nn.dropout(inputs, rate=self.rate, name="MCDropout")

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

# Normalize features
normalizer = Normalization()
normalizer.adapt(X_train)

# Model building
def build_model():
    model = Sequential([
        Input(shape=(X_poly.shape[1],)),
        normalizer,
        Dense(512, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Activation('relu'),
        MCDropout(0.5),
        Dense(512, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Activation('relu'),
        MCDropout(0.5),
        Dense(len(labels), activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy', AUC(name='auc')])
    return model

model = build_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Active Learning Loop
num_iterations = 5
samples_per_iter = 20

for iteration in range(num_iterations):
    model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping])
    predictions = np.array([model.predict(X_test) for _ in range(30)])
    mean_predictions = predictions.mean(axis=0)
    uncertainties = predictions.std(axis=0).mean(axis=1)

    uncertain_indices = np.argsort(uncertainties)[-samples_per_iter:]
    X_query, y_query = X_test[uncertain_indices], y_test[uncertain_indices]
    X_train = np.vstack([X_train, X_query])
    y_train = np.vstack([y_train, y_query])

    # Evaluate and report
    eval_metrics = model.evaluate(X_test, y_test)
    predicted_classes = np.round(mean_predictions)
    hamming_loss_value = hamming_loss(y_test, predicted_classes)
    report = classification_report(y_test, predicted_classes, target_names=labels, output_dict=True, zero_division=0)

    print(f"Iteration {iteration + 1}: Test Loss: {eval_metrics[0]}, Test Accuracy: {eval_metrics[1]}, Test AUC: {eval_metrics[2]}, Hamming Loss: {hamming_loss_value}")
    print(classification_report(y_test, predicted_classes, target_names=labels, zero_division=0))

# Plot ROC curves for each class
fpr, tpr, roc_auc = {}, {}, {}
plt.figure(figsize=(12, 10))
for i in range(len(labels)):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], mean_predictions[:, i])
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
errors = predictions.std(axis=0)[sample_index]
plt.figure(figsize=(12, 8))
plt.errorbar(range(len(labels)), mean_predictions[sample_index], yerr=errors, fmt='o', color='b', ecolor='r', capthick=2)
plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
plt.title('Predictions with Uncertainties for Each Species')
plt.xlabel('Species')
plt.ylabel('Predicted Probability')
plt.show()

