import torch
import sys


def check_cuda():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        # Print the number of available GPUs
        print(f"CUDA is available! \nNumber of GPUs: {torch.cuda.device_count()}")

        # Get the current CUDA device
        current_device = torch.cuda.current_device()

        # Get the name of the current device
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Current CUDA device: {current_device} ({device_name})")

    else:
        print("CUDA is not available!")

    print(f"Python version: {sys.version}")


if __name__ == "__main__":
    check_cuda()
