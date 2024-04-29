import torch
import pandas as pd
from iTransformer import iTransformer
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Load and preprocess the airline passengers dataset
data = pd.read_csv('data/AirPassengers.csv', index_col='Month', parse_dates=True)
data = data.iloc[:, 0].values  # Extract the passenger count column
data = (data - data.mean()) / data.std()  # Normalize the data

# Split the data into training and validation sets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
val_data = data[train_size:]

# Create input sequences and targets
def create_sequences(data, lookback_len, pred_length):
    X, y = [], []
    for i in range(len(data) - lookback_len - pred_length + 1):
        X.append(data[i:i+lookback_len])
        y.append(data[i+lookback_len:i+lookback_len+pred_length])
    return np.array(X), np.array(y)

lookback_len = 12
pred_length = 3

X_train, y_train = create_sequences(train_data, lookback_len, pred_length)
X_val, y_val = create_sequences(val_data, lookback_len, pred_length)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)

# Create an instance of the iTransformer model
model = iTransformer(
    num_variates=1,
    lookback_len=lookback_len,
    dim=64,
    depth=3,
    heads=4,
    dim_head=16,
    pred_length=pred_length
)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
batch_size = 32

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_X)
        loss = criterion(preds[pred_length][:, -1], batch_y[:, -1])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(X_val)
        val_loss = criterion(preds[pred_length][:, -1], y_val[:, -1])
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss.item():.4f}")