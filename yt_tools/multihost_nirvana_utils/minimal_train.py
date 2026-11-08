"""
This is a minimalistic accelerate training script used to test that distributed training works.
All rights belong to ChatGPT-4o
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator

# Initialize Accelerator
accelerator = Accelerator()

# Generate random dataset
input_size, hidden_size, output_size = 10, 20, 1
num_samples = 1000

X = torch.randn(num_samples, input_size)
y = torch.randn(num_samples, output_size)

# Create DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Model, Loss, Optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Prepare everything with Accelerator
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs, targets = batch

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

print("Training Completed!")
