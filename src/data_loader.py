import numpy as np
import pandas as pd

def load_data(path):
    import numpy as np
    import pandas as pd
    
    df = pd.read_csv(path)
    
    series = df.iloc[:, 1].values
    
    # Calculate mean and std
    mean = np.mean(series)
    std = np.std(series)
    
    # Normalize
    series = (series - mean) / std
    
    return series, mean, std   # ✅ IMPORTANT

def create_windows(data, window_size, horizon):
    X, y = [], []
    
    # WHY: Convert continuous sequence into supervised learning format
    for i in range(len(data) - window_size - horizon):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+horizon])
    
    return np.array(X), np.array(y)

def split_data(X, y):
    split = int(0.8 * len(X))
    
    # WHY: Chronological split preserves temporal order
    return X[:split], X[split:], y[:split], y[split:]