import torch

ckpt_path = "models/production/best_3class_legacy.pth"
checkpoint = torch.load(ckpt_path, map_location="cpu")

print("="*40)
print(f"DEBUGGING CHECKPOINT: {ckpt_path}")
print("="*40)

if "class_names" in checkpoint:
    print(f"REAL CLASS NAMES IN FILE: {checkpoint['class_names']}")
else:
    print("No class_names found in checkpoint metadata.")

# Look at the last layer weights to infer class count
if "model_state_dict" in checkpoint:
    sd = checkpoint["model_state_dict"]
    # Look for the last linear layer bias (usually has num_classes elements)
    for key in reversed(sd.keys()):
        if "bias" in key:
            print(f"Last Layer '{key}' size: {sd[key].shape}")
            break
print("="*40)
