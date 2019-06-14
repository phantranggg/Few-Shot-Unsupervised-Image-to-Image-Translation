import torch
import torch.nn as nn
import torchvision

class AdaptiveInstanceNormalization(nn.Module):
    def __init__(self):
        super(AdaptiveInstanceNormalization, self).__init__()
