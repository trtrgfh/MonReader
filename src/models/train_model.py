import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
import pickle

# Define the CNN model
# "SAME" padding = ceil((f - 1) / 2)
model = nn.Sequential(
  nn.Conv2d(3, 8, kernel_size=4, stride=1, padding=2),
  nn.ReLU(),
  nn.MaxPool2d(kernel_size=8, stride=8, padding=4),
  nn.Conv2d(8, 16, kernel_size=2, stride=1, padding=1),
  nn.ReLU(),
  nn.MaxPool2d(kernel_size=4, stride=4, padding=2),
  nn.Flatten(),
  nn.Linear(16 * 8 * 8, 6),
  nn.Softmax(dim=1)
)

torch.manual_seed(1234)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model.to(device)

for epoch in range(num_epochs):
  # Training
  train_loss = 0
  for images, labels in train_dataloader:
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)

    # Calculate loss
    loss = loss_fn(outputs, labels)
    train_loss += loss

    # Optimizer zero grad
    optimizer.zero_grad()

    # Loss backward
    loss.backward()

    # Optimizer step
    optimizer.step()

  train_loss /= len(train_dataloader)

  # Testing
  test_loss, test_acc = 0, 0
  final_y_pred = []
  model.eval()
  with torch.inference_mode():
    for images, labels in test_dataloader:
      images = images.to(device)
      labels = labels.to(device)

      # Forward pass
      outputs = model(images)

      # Calculate loss
      loss = loss_fn(outputs, labels)
      test_loss += loss
      test_acc += (torch.eq(labels, outputs.argmax(dim=1)).sum().item() / len(outputs)) * 100

      final_y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
    f1 = f1_score(test_labels, final_y_pred)

  print(f"Epoch: {epoch+1}")
  print(f"Train_loss: {train_loss}, Test_loss: {test_loss}, Test_acc: {test_acc}, f1: {f1}")
  
# Save trained best model
with open('cnn_model.pkl', 'wb') as file:
    pickle.dump(model, file)
