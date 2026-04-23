# Time Series Forecasting Pipeline

## Parameters (Roll No: 102303802)
- Window Size = 17
- Prediction Horizon = 3
- Hidden Size = 14

## Models Implemented
- MLP (Baseline)
- Custom RNN (from scratch)
- LSTM
- Transformer

## Key Concepts
- Sliding window converts sequence to supervised learning
- RNN uses hidden state as memory
- MLP ignores temporal structure

## Observations
- RNN captures sequence better than MLP
- Larger window improves context but increases complexity
- RNN struggles with long-term dependencies

## Evaluation Metrics
- MSE
- MAE
- RMSE