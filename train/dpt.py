import torch
import torch.nn as nn
# from transformers import AutoModelForDepthEstimation
from mlp import MLP
from depth_anything_3.api import DepthAnything3
from padding import ERPCircularPad2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DA360(nn.Module):
    def __init__(self, hf_model_id="depth-anything/DA3METRIC-LARGE"):
        super().__init__()
        
        # 1. Load the model
        base_wrapper = DepthAnything3.from_pretrained(hf_model_id)
        self.da3 = base_wrapper.model 
        
        # 2. Initialize our custom components
        embed_dim = 1024 
        self.mlp = MLP(embed_dim)
        self._replace_zero_padding_with_circular(self.da3.head)
        
        # 3. THE WIRETAP: Register a forward hook on the ViT's final normalization layer
        # This guarantees we capture the sequence right before the class token is dropped
        self._cls_token = None
        self.da3.backbone.pretrained.norm.register_forward_hook(self._capture_cls_token)

    def _capture_cls_token(self, module, input, output):
        # output shape is likely (Batch, Frames, Sequence, Embed_Dim)
        # Using [..., 0, :] grabs the 0th token of the Sequence dimension universally
        raw_cls = output[..., 0, :] 
        
        # raw_cls is now (Batch, Frames, Embed_Dim). 
        # We flatten it to strictly (Batch, Embed_Dim) so the MLP doesn't explode
        batch_size = raw_cls.shape[0]
        self._cls_token = raw_cls.view(batch_size, -1)

    def forward(self, x):
        # 1. SEQUENCE FORMATTING
        is_image = (x.dim() == 4)
        if is_image:
            x = x.unsqueeze(1)
            
        # 2. THE HEAVY LIFTING
        output = self.da3(x) 
        
        # 3. EXTRACT TENSOR
        if isinstance(output, dict):
            affine_disparity = output.get('depth', output.get('pred', list(output.values())[0]))
        else:
            affine_disparity = output
            
        # 4. SHAPE CLEANUP (Disparity)
        if is_image and affine_disparity.dim() == 5:
            affine_disparity = affine_disparity.squeeze(1) 
            
        # Ensure it is explicitly (Batch, 1, H, W)
        if affine_disparity.dim() == 3:
            affine_disparity = affine_disparity.unsqueeze(1)
        
        # 5. SHIFT MLP [cite: 154, 155]
        cls_token = self._cls_token
        shift = self.mlp(cls_token) # GUARANTEED shape: (Batch, 1)
        
        # 6. SCALE INVARIANCE [cite: 149]
        # Force shift to explicitly match the 4D disparity map: (Batch, 1, 1, 1)
        shift = shift.view(-1, 1, 1, 1)
        
        scale_invariant_disparity = affine_disparity + shift
        
        return scale_invariant_disparity

    def _replace_zero_padding_with_circular(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d) and child.padding != (0, 0) and child.padding != 0:
                # 1. Save the original padding value
                pad_val = child.padding
                
                # 2. Mutate the convolution to have NO padding (0, 0)
                child.padding = (0, 0)
                
                # 3. Wrap it in a Sequential: ERPCircularPad2d -> Conv2d (without padding)
                setattr(module, name, nn.Sequential(
                    ERPCircularPad2d(pad_val),
                    child
                ))
            else:
                # Recursively search deeper into nested modules
                self._replace_zero_padding_with_circular(child)