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


class identify_pass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        #ctx.save_for_backward(input)
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        torch.save(grad_output, os.path.join('./ref_loss/proto_loss.pth'))
        return grad_output
    

class proto_bypass_act(Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        
    def _bypass_forward(self, input):
        return identify_pass.apply(input)
        
    def forward(self, input: Tensor) -> Tensor:
        return self._bypass_forward(input)


class seg_pass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        #ctx.save_for_backward(input)
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        torch.save(grad_output, os.path.join('./ref_loss/seg_loss.pth'))
        return grad_output
    

class seg_bypass_act(Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        
    def _bypass_forward(self, input):
        return seg_pass.apply(input)
        
    def forward(self, input: Tensor) -> Tensor:
        return self._bypass_forward(input)


class pred_identify_pass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        #ctx.save_for_backward(input)
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        if grad_output.shape[1] == 75:
            pred_shape = 5
        elif grad_output.shape[1] == 243:
            pred_shape = 9
        elif grad_output.shape[1] == 972:
            pred_shape = 18
        elif grad_output.shape[1] == 3675:
            pred_shape = 35
        elif grad_output.shape[1] == 14283:
            pred_shape = 69

        if grad_output.shape[2]==4:
            bbox_loss = grad_output.reshape(grad_output.shape[0], pred_shape, pred_shape, -1).permute(0,3,1,2)
            # print('bbox_loss :', bbox_loss.shape, grad_output.abs().sum())
            torch.save(bbox_loss, os.path.join('./ref_loss/', str(bbox_loss.shape[2])+'_bbox_loss.pth'))
        elif grad_output.shape[2]==21:
            conf_loss = grad_output.reshape(grad_output.shape[0], pred_shape, pred_shape, -1).permute(0,3,1,2)
            # print('conf_loss :', conf_loss.shape, grad_output.abs().sum())
            torch.save(conf_loss, os.path.join('./ref_loss/', str(conf_loss.shape[2])+'_conf_loss.pth'))
        elif grad_output.shape[2]==32:
            mask_loss = grad_output.reshape(grad_output.shape[0], pred_shape, pred_shape, -1).permute(0,3,1,2)
            # print('mask_loss :', mask_loss.shape, grad_output.abs().sum())
            torch.save(mask_loss, os.path.join('./ref_loss/', str(mask_loss.shape[2])+'_mask_loss.pth'))
        return grad_output


class pred_bypass_act(Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        
    def _bypass_forward(self, input):
        return pred_identify_pass.apply(input)
        
    def forward(self, input: Tensor) -> Tensor:
        return self._bypass_forward(input)