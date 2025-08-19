import torch
import random
import logging
import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    "plot_losses",
    "get_device",
    "set_seed"
]


# ------------------ Visualization ------------------ #
def plot_losses(statistics: dict[str, list[float]]):
    """
    Plots the training and validation loss on the same graph for direct comparison.

    Args:
        statistics (dict): A dictionary with two keys:
            - 'train' (list): Training loss values per epoch.
            - 'validation' (list): Validation loss values per epoch.

    The function creates a single plot:
    - The x-axis represents epochs.
    - The y-axis represents the loss values.
    - Both train and validation losses are plotted with different colors and markers.

    Raises:
        ValueError: If `statistics` doesn't hold 'train' or 'validation'.
    """
    if "train" not in statistics or "validation" not in statistics:
        logging.error("Input dictionary must contain 'train' and 'validation' keys "
                      "for _plot_losses.")
        raise ValueError("Input dictionary must contain 'train' and 'validation' "
                         "keys.")
    # --- Data Extraction ---
    train_loss = statistics['train']
    validation_loss = statistics['validation']
    epochs = range(1, len(train_loss) + 1)
    # --- Plotting Configuration ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, linestyle='-', color='#1f77b4',
             label='Train Loss', linewidth=2)
    plt.plot(epochs, validation_loss, linestyle='-', color='#d62728',
             label='Validation Loss', linewidth=2)
    # --- Chart Customization ---
    plt.title("Training & Validation Loss Over Epochs",
              fontsize=18, fontweight='bold')
    plt.xticks(epochs) # This ensures that xticks are integers
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    # --- Display Plot ---
    plt.show()


# -------------- Device Configuration --------------- #

def get_device():
    """
    Selects and returns the optimal device (GPU or CPU) for computation.

    This function first checks for the availability of a NVIDIA GPU with
    CUDA support. If a GPU is found, it's chosen as the computation device.
    Otherwise, it defaults to the CPU. A descriptive message is printed to
    inform the user which device has been selected. This helps in verifying
    that the hardware is correctly recognized for accelerated computations.

    Returns:
        torch.device: The selected device, either 'cuda' or 'cpu'.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(device)
        print(f"Using GPU: {device_name}\n")
    else:
        device = torch.device('cpu')
        print("Using CPU\n")

    return device


def set_seed(seed_value: int = 73):
    """
    Sets the random seed for reproducibility across multiple libraries.

    This function ensures that the random number generators in Python's
    built-in `random` module, NumPy, and PyTorch are all initialized
    with the same seed. This is crucial for creating reproducible
    experiments in machine learning, as it guarantees that operations
    involving randomness (like data shuffling, weight initialization,
    and dropout) will yield the same results every time the code is run.

    Args:
        seed_value (int): The integer value to use as the seed. Defaults to 73.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    # If a GPU is available, set the seed for all CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
