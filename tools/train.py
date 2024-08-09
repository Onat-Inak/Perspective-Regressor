import wandb
from tqdm.notebook import tqdm
from tools.validate import validate

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "last_expr"


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, config):
    n_total_steps = len(train_loader)
    running_train_loss = 0.0
    average_train_loss = 0.0
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=config.log_period)

    # Run training and track with wandb
    seen_images = 0  # number of examples seen
    current_batch = 0
    for epoch in tqdm(range(config.num_epochs)):
        
        # Set model to training mode
        model.train()
        for batch, (X_, y_) in enumerate(train_loader):
            seen_images += X_.size(0)
            current_batch += 1

            if epoch == 0 and batch == 0:
                model.eval()
                val_loss = validate(model, val_loader, config.device, criterion)
                model.train()
                wandb.log(
                    {
                        "val_loss": val_loss,
                        "epoch": epoch + 1,
                    }
                )

            # Calculate training loss in the corresponding batch
            train_loss = train_batch(X_, y_, model, optimizer, criterion, config)
            running_train_loss += train_loss.item()

            # Log first training loss
            if epoch == 0 and batch == 0:
                wandb.log({"train_loss": train_loss})

            if (batch + 1) % config.log_period == 0:
                average_train_loss = running_train_loss / config.log_period
                print(
                    f"Epoch [{epoch+1}/{config.num_epochs}], Step [{batch+1}/{n_total_steps}], Current Batch [{current_batch}/{n_total_steps * config.num_epochs}], Train Loss: {average_train_loss:.4f}"
                )
                # Log training metrics
                train_log(
                    average_train_loss,
                    current_batch,
                    epoch,
                    optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[1]["lr"],
                )
                running_train_loss, average_train_loss = 0.0, 0.0
        # Calculate validation accuracy
        model.eval()
        val_loss = validate(model, val_loader, config.device, criterion)
        model.train()
        wandb.log(
            {
                "val_loss": val_loss,
                "epoch": epoch + 1,
            }
        )
        print(f"Validation Loss: {val_loss:.5f} \n")
        
        # Decrease the learning rate regarding config.gamma
        scheduler.step(val_loss)


def train_batch(X_, y_, model, optimizer, criterion, config):
    X_, y_ = X_.to(config.device), y_.to(config.device)
    
    # Forward pass
    outputs = model(X_)
    train_loss = criterion(outputs.to(config.device), y_)

    # Backward pass
    optimizer.zero_grad()
    train_loss.backward()

    # Step with optimizer
    optimizer.step()

    return train_loss


def train_log(train_loss, current_batch, epoch, backbone_lr, other_lr):
    # Log to W&B
    wandb.log(
        {"epoch": epoch + 1, "train_loss": train_loss, "backbone_lr": backbone_lr, "other_lr": other_lr, "current_batch": current_batch},
        step=current_batch,
    )
    print("backbone_lr: ", backbone_lr, "other_lr: ", other_lr)
