import torch
import torch.nn as nn
import torch.optim as optim
from dpt import DA360
from scale_invariance import ScaleInvariantLoss

# (Assume your DA360 class, ERPCircularPad2d, and MLP are defined here)

def test_da360_pipeline():
    print("1. Instantiating the DA360 model...")
    # Note: This might take a moment to download/load the HF weights
    model = DA360(hf_model_id="depth-anything/da3metric-large")
    loss_func = ScaleInvariantLoss()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train() # Put model in training mode

    print("2. Creating dummy tensors...")
    # The DA360 paper explicitly uses a resolution of 518x1036 for training and eval [cite: 244]
    batch_size = 1
    dummy_image = torch.randn(batch_size, 3, 518, 1036).to(device)
    
    # Fake ground truth disparity map (e.g., from ProcTHOR10k)
    dummy_ground_truth = torch.randn(batch_size, 1, 518, 1036).to(device)

    print("3. Setting up the Optimizer...")
    # The paper uses AdamW with a peak learning rate of 1e-5 
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    print("4. Executing Forward Pass with AMP...")
    optimizer.zero_grad() 
    
    # Use AMP to cut memory usage in half
    scaler = torch.amp.GradScaler('cuda')
    
    with torch.amp.autocast('cuda'):
        predicted_disparity = model(dummy_image)
        # (We will swap MSE for the Scale-Invariant Loss here shortly)
        loss = loss_func.forward(predicted_disparity, dummy_ground_truth)
        
    print(f"   Loss value: {loss.item():.4f}")

    print("5. Executing Backward Pass (Backprop)...")
    # Use the scaler to backpropagate safely with 16-bit floats
    scaler.scale(loss).backward()
    
    print(f"   Shift MLP gradients: {model.mlp.mlp[0].weight.grad is not None}")

    print("6. Updating Weights...")
    scaler.step(optimizer)
    scaler.update()
    
    print("Pipeline test complete!")

# Run the test
if __name__ == "__main__":
    test_da360_pipeline()