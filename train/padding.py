import torch
import torch.nn.functional as F
import torch.nn as nn


class ERPCircularPad2d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        # Handle the different ways padding might be passed
        if isinstance(padding, int):
            self.pad_l = self.pad_r = self.pad_t = self.pad_b = padding
        elif isinstance(padding, (tuple, list)) and len(padding) == 2:
            # Conv2d padding is (pad_height, pad_width)
            pad_h, pad_w = padding
            self.pad_t = self.pad_b = pad_h
            self.pad_l = self.pad_r = pad_w
        elif isinstance(padding, (tuple, list)) and len(padding) == 4:
            # Explicit (left, right, top, bottom)
            self.pad_l, self.pad_r, self.pad_t, self.pad_b = padding
        else:
            raise ValueError(f"Unsupported padding format: {padding}")

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Vertical Phase [cite: 168, 169]
        if self.pad_t > 0 or self.pad_b > 0:
            x_padded_v = torch.zeros((B, C, H + self.pad_t + self.pad_b, W), device=x.device, dtype=x.dtype)
            x_padded_v[:, :, self.pad_t : self.pad_t + H, :] = x
            
            if self.pad_t > 0:
                top_strip = x[:, :, :self.pad_t, :]
                top_strip_flipped = torch.flip(top_strip, dims=[2])
                top_strip_rolled = torch.roll(top_strip_flipped, shifts=W//2, dims=3)
                x_padded_v[:, :, :self.pad_t, :] = top_strip_rolled
                
            if self.pad_b > 0:
                bottom_strip = x[:, :, -self.pad_b:, :]
                bottom_strip_flipped = torch.flip(bottom_strip, dims=[2])
                bottom_strip_rolled = torch.roll(bottom_strip_flipped, shifts=W//2, dims=3)
                x_padded_v[:, :, -self.pad_b:, :] = bottom_strip_rolled
                
            x = x_padded_v

        # 2. Horizontal Phase [cite: 168, 170]
        if self.pad_l > 0 or self.pad_r > 0:
            x = F.pad(x, (self.pad_l, self.pad_r, 0, 0), mode='circular')
            
        return x