import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import math
# -*- coding: utf-8 -*-
import math
import warnings
import os

import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
from torch.nn import Module

from typing import Optional, List, Tuple, Union


class custom_ReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        #print('ProtoNet ReLU Loss :', grad_input.shape, grad_input.abs().sum())
        torch.save(grad_input, os.path.join('./ref_loss/proto_loss.pth'))

        return grad_input


class customReLU(Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        
    def _relu_forward(self, input):
        return custom_ReLU.apply(input)
        
    def forward(self, input: Tensor) -> Tensor:
        return self._relu_forward(input)