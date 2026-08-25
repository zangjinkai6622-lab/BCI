import torch
import torch.nn as nn
from dataset import train_loader
from model import model


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(10):
    total_loss = 0
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for x, y in train_loader:
            prediction = model(x)
            total_loss += loss_fn(prediction, y)

    average_loss = total_loss / len(train_loader)

    print(
        "Epoch:",
        epoch,
        "Average Loss:",
        average_loss
    )