import torch
import torch.nn as nn
import dataset
from model import model


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam( #Adam会根据历史gradient自动调整不同参数的更新幅度，公式复杂
    model.parameters(),
    lr=0.001,  #learniong_rate 学习率,Learning Rate 决定参数每次更新的步长
    weight_decay=1e-4 #限制参数过度变大，让模型不要过度依赖训练数据，一种正则化手段
)
# optimizer.step()背后就是根据gradient和lr计算新的w，比如SGD就是w=w-lr*graident

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( #Scheduler：调整 Optimizer 使用的 Learning Rate。
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