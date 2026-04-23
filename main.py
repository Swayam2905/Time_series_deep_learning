import torch
import matplotlib.pyplot as plt

from src.data_loader import load_data, create_windows, split_data
from src.models import MLP, CustomRNN, LSTMModel, TransformerModel
from src.train import train_model
from src.evaluate import evaluate
from src.ablation import run_ablation

# ================================
# PARAMETERS (FROM ROLL NUMBER)
# ================================
window_size = 17
prediction_horizon = 3
hidden_size = 14

print(window_size, prediction_horizon, hidden_size)

# ================================
# LOAD DATASET 1
# ================================
series, mean, std = load_data("data/electricity.csv")

X, y = create_windows(series, window_size, prediction_horizon)
X_tr, X_te, y_tr, y_te = split_data(X, y)

X_tr = torch.tensor(X_tr, dtype=torch.float32)
y_tr = torch.tensor(y_tr, dtype=torch.float32)
X_te = torch.tensor(X_te, dtype=torch.float32)
y_te = torch.tensor(y_te, dtype=torch.float32)

# ================================
# MODELS
# ================================
mlp = MLP(window_size, prediction_horizon)
rnn = CustomRNN(hidden_size, prediction_horizon)
lstm = LSTMModel(hidden_size, prediction_horizon)
transformer = TransformerModel(hidden_size, prediction_horizon)

# ================================
# TRAIN
# ================================
print("\nTraining MLP")
mlp_loss = train_model(mlp, X_tr, y_tr)

print("\nTraining RNN")
rnn_loss = train_model(rnn, X_tr, y_tr)

print("\nTraining LSTM")
lstm_loss = train_model(lstm, X_tr, y_tr)

print("\nTraining Transformer")
trans_loss = train_model(transformer, X_tr, y_tr)

# ================================
# EVALUATE
# ================================
models = {
    "MLP": mlp,
    "RNN": rnn,
    "LSTM": lstm,
    "Transformer": transformer
}

for name, model in models.items():
    preds, mse, mae, rmse = evaluate(model, X_te, y_te, mean, std)
    print(f"{name}: MSE={mse}, MAE={mae}, RMSE={rmse}")

# ================================
# PLOTS
# ================================
plt.plot(rnn_loss)
plt.title("RNN Training Loss")
plt.show()

preds, _, _, _ = evaluate(rnn, X_te, y_te, mean, std)

plt.plot(y_te.numpy()[:, 0], label="Actual")
plt.plot(preds[:, 0], label="Predicted")
plt.legend()
plt.title("Prediction vs Actual")
plt.show()

# ================================
# ABLATION
# ================================
run_ablation(series, window_size, prediction_horizon, hidden_size)


print("\n==============================")
print("RUNNING ON SECOND DATASET")
print("==============================")

# Load second dataset
series2, mean2, std2 = load_data("data/second_dataset.csv")

X2, y2 = create_windows(series2, window_size, prediction_horizon)
X2_tr, X2_te, y2_tr, y2_te = split_data(X2, y2)

X2_tr = torch.tensor(X2_tr, dtype=torch.float32)
y2_tr = torch.tensor(y2_tr, dtype=torch.float32)
X2_te = torch.tensor(X2_te, dtype=torch.float32)
y2_te = torch.tensor(y2_te, dtype=torch.float32)

# Train ONLY RNN on second dataset
rnn2 = CustomRNN(hidden_size, prediction_horizon)

print("\nTraining RNN on second dataset...")
train_model(rnn2, X2_tr, y2_tr, epochs=20)

# Evaluate
preds2, mse2, mae2, rmse2 = evaluate(rnn2, X2_te, y2_te, mean2, std2)

print(f"\nSecond Dataset Results:")
print(f"MSE={mse2}, MAE={mae2}, RMSE={rmse2}")

import matplotlib.pyplot as plt

plt.plot(y2_te.numpy()[:100, 0], label="Actual")
plt.plot(preds2[:100, 0], label="Predicted")
plt.legend()
plt.title("Second Dataset Prediction vs Actual")
plt.show()