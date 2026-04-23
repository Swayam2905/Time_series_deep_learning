import torch
import torch.nn as nn

# ================================
# MLP BASELINE
# ================================
class MLP(nn.Module):
    def __init__(self, window_size, horizon):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(window_size, 64),
            nn.ReLU(),
            nn.Linear(64, horizon)
        )
    
    def forward(self, x):
        # WHY: MLP treats input as flat vector (no sequence awareness)
        return self.model(x)


# ================================
# CUSTOM RNN (FROM SCRATCH)
# ================================
class CustomRNN(nn.Module):
    def __init__(self, hidden_size, horizon):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.Wx = nn.Linear(1, hidden_size)
        self.Wh = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        batch_size = x.size(0)
        
        # WHY: Hidden state stores memory of previous inputs
        h = torch.zeros(batch_size, self.hidden_size)

        for t in range(x.size(1)):
            x_t = x[:, t].unsqueeze(1)
            
            # WHY: Combine current input + previous memory
            h = torch.tanh(self.Wx(x_t) + self.Wh(h))

        return self.fc(h)


# ================================
# LSTM (PREBUILT)
# ================================
class LSTMModel(nn.Module):
    def __init__(self, hidden_size, horizon):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        x = x.unsqueeze(-1)
        
        # WHY: LSTM handles long-term dependencies better than RNN
        _, (h, _) = self.lstm(x)
        
        return self.fc(h[-1])


# ================================
# TRANSFORMER
# ================================
class TransformerModel(nn.Module):
    def __init__(self, hidden_size, horizon):
        super().__init__()
        
        self.embedding = nn.Linear(1, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=2,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        
        # WHY: Transformer uses attention instead of sequential memory
        x = self.transformer(x)
        
        return self.fc(x[:, -1, :])