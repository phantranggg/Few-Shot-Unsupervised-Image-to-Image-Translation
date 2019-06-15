import torch
import torch.nn as nn
import torchvision
from models.resBlk import ResBlocks
from models.adaIN import AdaptiveInstanceNormalization


class Encoder(nn.Module):
    def __init__(self, in_planes, plane):
        super(Encoder, self).__init__()

