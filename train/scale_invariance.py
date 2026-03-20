import torch
import torch.nn as nn

class ScaleInvariantLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target, mask=None):
        if mask is None:
            mask = torch.ones_like(prediction, dtype=torch.bool)
            
        loss = 0.0
        batch_size = prediction.size(0)
        
        for i in range(batch_size):
            pred_i = prediction[i][mask[i]]
            targ_i = target[i][mask[i]]
            
            if pred_i.numel() == 0:
                continue
                
            # 1. Robust shift (median)
            t_pred = torch.median(pred_i)
            t_targ = torch.median(targ_i)
            
            # 2. Robust scale (mean absolute deviation)
            s_pred = torch.mean(torch.abs(pred_i - t_pred))
            s_targ = torch.mean(torch.abs(targ_i - t_targ))
            
            s_pred = torch.clamp(s_pred, min=1e-8)
            s_targ = torch.clamp(s_targ, min=1e-8)
            
            # 3. Normalize
            pred_normalized = (pred_i - t_pred) / s_pred
            targ_normalized = (targ_i - t_targ) / s_targ
            
            # 4. Compute Loss
            loss += torch.mean(torch.abs(pred_normalized - targ_normalized))
            
        return loss / batch_size