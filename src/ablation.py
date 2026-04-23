import torch
from src.data_loader import create_windows, split_data
from src.train import train_model
from src.evaluate import evaluate
from src.models import CustomRNN

def run_ablation(series, window_size, horizon, hidden_size):
    
    print("\nAblation Study:")

    for w in [window_size, window_size//2, window_size*2]:
        
        X, y = create_windows(series, w, horizon)
        X_tr, X_te, y_tr, y_te = split_data(X, y)
        
        X_tr = torch.tensor(X_tr, dtype=torch.float32)
        y_tr = torch.tensor(y_tr, dtype=torch.float32)
        X_te = torch.tensor(X_te, dtype=torch.float32)
        y_te = torch.tensor(y_te, dtype=torch.float32)

        model = CustomRNN(hidden_size, horizon)
        
        train_model(model, X_tr, y_tr, epochs=10)
        
        _, mse, _, _ = evaluate(model, X_te, y_te)
        
        print(f"Window {w} → MSE: {mse}")