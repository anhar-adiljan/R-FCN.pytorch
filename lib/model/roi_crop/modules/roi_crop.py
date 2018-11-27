import torch.nn as nn
from ..functions.roi_crop import RoICropFunction

class _RoICrop(nn.Module):
    def __init__(self, layout = 'BHWD'):
        super(_RoICrop, self).__init__()

    def forward(self, input1, input2):
        return RoICropFunction()(input1, input2)
