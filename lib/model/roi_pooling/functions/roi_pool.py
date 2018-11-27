import torch
from torch.autograd import Function
from .._ext import roi_pooling
import pdb

class RoIPoolFunction(Function):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        self.pooled_width = pooled_width
        self.pooled_height = pooled_height
        self.spatial_scale = spatial_scale
        self.feature_size = None

    def forward(self, features, rois): 
        self.feature_size = features.size()           
        batch_size, num_channels, data_height, data_width = self.feature_size
        num_rois = rois.size(0)

        output_size = (num_rois, num_channels, self.pooled_height, self.pooled_width)
        output = features.new_zeros(output_size)
        self.argmax = features.new_zeros(output_size, dtype=torch.int)

        self.rois = rois
        if not features.is_cuda:
            _features = features.permute(0, 2, 3, 1)
            roi_pooling.roi_pooling_forward(self.pooled_height, self.pooled_width, self.spatial_scale,
                                            _features, rois, output)
        else:
            roi_pooling.roi_pooling_forward_cuda(self.pooled_height, self.pooled_width, self.spatial_scale,
                                                 features, rois, output, self.argmax)

        return output

    def backward(self, grad_output):
        assert(self.feature_size is not None and grad_output.is_cuda)
        batch_size, num_channels, data_height, data_width = self.feature_size
        grad_input = grad_output.new_zeros((batch_size, num_channels, data_height, data_width))

        roi_pooling.roi_pooling_backward_cuda(self.pooled_height, self.pooled_width, self.spatial_scale,
                                              grad_output, self.rois, grad_input, self.argmax)

        return grad_input, None
