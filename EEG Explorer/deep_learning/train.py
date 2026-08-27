import torch
import torch.nn as nn
import dataset
from model import model


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50
)

best_val_loss = float("inf")

for epoch in range(50):
    train_loss = 0
    train_correct = 0
    train_total = 0

    model.train()

    for x, y in dataset.train_loader:
        y = y.squeeze()
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
        train_correct += (
            logits.argmax(dim=1) == y
        ).sum().item()
        train_total += x.size(0)
    val_loss = 0
    val_correct = 0
    val_total = 0


    model.eval()
    with torch.no_grad():
        for x, y in dataset.val_loader:
            y = y.squeeze()
            logits = model(x)
            loss = loss_fn(logits, y)
            val_loss += loss.item() * x.size(0)
            val_correct += (
                logits.argmax(dim=1) == y
            ).sum().item()

            val_total += x.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total
    val_loss /= val_total
    val_acc = val_correct / val_total
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            model.state_dict(),
            "best_model.pth"
        )
        best_mark = " ← Best"
    else:
        best_mark = ""

    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch [{epoch + 1:3d}/50] | "
        f"LR: {current_lr:.6f} | "
        f"Train Loss: {train_loss:.4f} "
        f"Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} "
        f"Acc: {val_acc:.4f}"
        f"{best_mark}"
    )