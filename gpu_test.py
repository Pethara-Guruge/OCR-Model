import torch
import torch_directml

# Initialize DirectML
dml = torch_directml.device()  # Gets default DirectML device

# Create tensor and move to GPU
tensor = torch.randn(3, 3).to(dml)  # Correct way to transfer tensors

print(f"Device count: {torch_directml.device_count()}")  # Should be 1
print(f"Using device: {dml}")  # Shows 'privateuseone:0'
print(f"Tensor device: {tensor.device}")  # Should match dml