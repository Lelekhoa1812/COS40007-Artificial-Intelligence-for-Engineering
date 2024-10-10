# THIS DOES NOT RUN ON THIS MACHINE ANYWAY AS NOT CAPABLE TO SWITCH TO GPU SINCE NVIDIA GPU IS REQUIRED TO USE PYTORCH ON CPU RUNTIME
import torch

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # If cuda is not available means that we are using CPU
print(f"Using {device} for training")

# Simple test
tensor = torch.randn(3,3) # Tensor on CPU
tensor= tensor.to('cuda') # Move to GPU
print(tensor) # Should print 'cuda' if the GPU is available and 'cpu' otherwise