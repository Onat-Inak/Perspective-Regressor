import wandb
import torch


def validate(model, val_loader, device, criterion):
    with torch.no_grad():
        val_loss = 0.0
        total_y_ = torch.tensor([]).to(device)
        total_outputs = torch.tensor([]).to(device)
        total = 0
        for batch, (X_, y_) in enumerate(val_loader):
            X_, y_ = X_.to(device), y_.to(device)
            outputs = model(X_)
            total_y_ = torch.cat((total_y_, y_), dim=0)
            total_outputs = torch.cat((total_outputs, outputs))
            total += y_.size(0)

        # Calculate regression metrics
        val_loss = criterion(total_outputs, total_y_).item()

    return val_loss
