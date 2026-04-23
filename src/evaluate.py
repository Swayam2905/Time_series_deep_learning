import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate(model, X_test, y_test, mean, std):
    model.eval()
    
    with torch.no_grad():
        preds = model(X_test).numpy()
    
    y_true = y_test.numpy()

    mse = mean_squared_error(y_true, preds)
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mse)
    
    return preds, mse, mae, rmse