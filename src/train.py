import torch
import torch.nn as nn

def train_model(model, X_train, y_train, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    losses = []

    for epoch in range(epochs):
        model.train()
        
        optimizer.zero_grad()
        
        output = model(X_train)
        
        # WHY: MSE penalizes large prediction errors
        loss = loss_fn(output, y_train)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return losses