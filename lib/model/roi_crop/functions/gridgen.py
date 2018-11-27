# functions/add.py
import torch
from torch.autograd import Function
import numpy as np


class AffineGridGenFunction(Function):
    def __init__(self, height, width,lr=1):
        super(AffineGridGenFunction, self).__init__()
        self.lr = lr
        self.height, self.width = height, width
        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/(self.height)), 0), repeats = self.width, axis = 0).T, 0)
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/(self.width)), 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.input1 = input1
        output = input1.new_zeros(torch.Size([input1.size(0)]) + self.grid.size())
        self.batchgrid = input1.new_zeros(torch.Size([input1.size(0)]) + self.grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid.type_as(self.batchgrid[i])

        for i in range(input1.size(0)):
            output = torch.bmm(self.batchgrid.view(-1, self.height*self.width, 3), torch.transpose(input1, 1, 2)).view(-1, self.height, self.width, 2)

        return output

    def backward(self, grad_output):

        grad_input1 = self.input1.new_zeros(self.input1.size())

        # if grad_output.is_cuda:
        #    self.batchgrid = self.batchgrid.cuda()
        #    grad_input1 = grad_input1.cuda()

        grad_input1 = torch.baddbmm(grad_input1, torch.transpose(grad_output.view(-1, self.height*self.width, 2), 1,2), self.batchgrid.view(-1, self.height*self.width, 3))
        return grad_input1
