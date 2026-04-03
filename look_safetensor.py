file_path = "outputs/train/dev_task1/rgb_depth_act/run_20251119_094752/policy_preprocessor_step_3_normalizer_processor.safetensors"

from safetensors import safe_open
import torch

# Open safetensor file

with safe_open(file_path, framework="pt", device="cpu") as f:
    # Get all tensor names
    tensor_names = f.keys()
    print("All tensor names:")
    for name in tensor_names:
        print(f"- {name}")
    print("\nTensor details:")
    # View details of each tensor
    for name in tensor_names:
        if name in ["action.max","action.min","action.mean","action.std"]:
            tensor = f.get_tensor(name)
            print(f"Name: {name}")
            print(f"Shape: {tensor.shape}")
            print(f"Data type: {tensor.dtype}")
            print(f"Value range: [{tensor.min():.6f}, {tensor.max():.6f}]")
            print(f"Mean: {tensor.mean():.6f}")
            
            # For small tensors, you can print the value directly
            if tensor.numel() <= 50:  # The number of elements is less than or equal to 10
                print(f"Value: {tensor}")
            print("-" * 50)