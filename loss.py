import torch
import torch.nn as nn

class JaccardLoss(nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()

    def forward(self, outputs, targets, smooth=1e-6):
        # Flatten the tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # Calculate intersection and union
        intersection = (outputs * targets).sum()
        total = (outputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)
        return 1 - IoU