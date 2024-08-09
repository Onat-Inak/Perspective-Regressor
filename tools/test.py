import wandb
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


def test(model, test_loader, device, project_name, save_model=False):
    model.eval()

    with torch.no_grad():
        total_y_ = torch.tensor([]).to(device)
        total_outputs = torch.tensor([]).to(device)
        total = 0
        for batch, (X_, y_) in enumerate(test_loader):
            X_, y_ = X_.to(device), y_.to(device)
            outputs = model(X_)
            total_y_ = torch.cat((total_y_, y_), dim=0)
            total_outputs = torch.cat((total_outputs, outputs))
            total += y_.size(0)

        # Convert tensors to numpy arrays for metric calculations
        total_y_ = total_y_.cpu().numpy()
        total_outputs = total_outputs.cpu().numpy()

        # Calculate regression metrics
        mae = mean_absolute_error(total_y_, total_outputs)
        mse = mean_squared_error(total_y_, total_outputs)

        # Print and log the results
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")

        # Log the results to wandb
        wandb.log({"MAE": mae, "MSE": mse})

    if save_model:
        # Save the model in the exchangeable ONNX format
        torch.save(
            model.state_dict(), "experiments/" + project_name + "/model_state_dict.pth"
        )
        # torch.save(model, 'experiments/RNN/model.pt')
        # wandb.unwatch()
        wandb.save("experiments/" + project_name + "/model_state_dict.pth")
        # wandb.save('experiments/RNN/model.pt')
