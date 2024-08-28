import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import math
# -*- coding: utf-8 -*-
import math
import warnings

import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
from torch.nn import Module


from typing import Optional, List, Tuple, Union


class custom_Conv2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, weight, bias=True, stride=1, padding=0, dilation=1, groups=1):
        ctx.save_for_backward(X, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        bits = 8
        
        #Quantization
        # weight_quant = weight_quantization(weight, bits)
        # act_quant = act_quantization(X, bits)
        # #Convolution
        # out = F.conv2d(act_quant, weight_quant, bias, stride, padding, dilation)
        
        out = F.conv2d(X, weight, bias, stride, padding, dilation)

        # Quantization
        # weight_quant, wt_scale = weight_quantizer(weight, bits)
        # act_quant, act_scale = act_quantizer(X, bits)
        # Convolution
        # out = F.conv2d(act_quant, weight_quant, None, stride, padding, dilation)
        # out = convolution(act_quant.type(torch.cuda.IntTensor), weight_quant.type(torch.cuda.IntTensor), stride = stride, pad_size = padding)
        # Dequantization
        # out = dequantizer(out, wt_scale, act_scale)
        # out += bias.reshape(1, -1, 1, 1)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_X = grad_w = grad_b = None
        input, weight, bias = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        if ctx.needs_input_grad[0]:
            #grad_X = F.conv_transpose2d(grad_output, weight, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
            grad_X = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)          
            
        if ctx.needs_input_grad[1]:
            #grad_w = F.conv2d(X.transpose(0, 1), grad_output.transpose(0, 1), bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
            #grad_w = grad_w.transpose(0, 1)
            grad_w = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_b = grad_output.sum(dim=(0, 2, 3))
        
        if grad_output.shape[1]==12:
            #print('bbox_loss :', grad_output.shape, grad_output.abs().sum())
            torch.save(grad_output, os.path.join('./ref_loss/', str(grad_output.shape[2])+'_bbox_loss.pth'))
        elif grad_output.shape[1]==63:
            #print('conf_loss :', grad_output.shape, grad_output.abs().sum())
            torch.save(grad_output, os.path.join('./ref_loss/', str(grad_output.shape[2])+'_conf_loss.pth'))
        elif grad_output.shape[1]==96:
            #print('mask_loss :', grad_output.shape, grad_output.abs().sum())
            torch.save(grad_output, os.path.join('./ref_loss/', str(grad_output.shape[2])+'_mask_loss.pth'))
        elif grad_output.shape[1]==20:
            #print('seg_loss :', grad_output.shape, grad_output.abs().sum())
            torch.save(grad_output, os.path.join('./ref_loss/seg_loss.pth'))

        return grad_X, grad_w, grad_b, None, None, None, None
        
class customConv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, bias = True, groups=1, device=None, dtype=None) -> None:
        super(customConv2d, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = Parameter(torch.empty(
            (out_channels, in_channels, kernel_size, kernel_size), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        
    def load(self, weight):
        self.weight = Parameter(weight[0].type(torch.cuda.FloatTensor), requires_grad=True)
        
        if len(weight) > 1:
            self.bias = Parameter(weight[1].type(torch.cuda.FloatTensor), requires_grad=True)


    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        return custom_Conv2d.apply(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)


def weight_quantizer(weight, bits):
    max_val = torch.max(weight)
    min_val = torch.min(weight)

    bits = math.pow(2, bits - 1)
    scaling_factor = bits/(max_val - min_val)
    weight_quant = torch.clamp(torch.round(weight * scaling_factor), -bits, bits - 1).type(torch.cuda.CharTensor)
    # weight_quant = RNTB(weight * scaling_factor)
    # weight_quant = torch.clamp(weight_quant, -bits, bits - 1)
    # weight_dequant = weight_quant / scaling_factor
    return weight_quant, scaling_factor

def act_quantizer(x,bits):
    max_val = torch.max(x)
    min_val = torch.min(x)

    bits = math.pow(2, bits - 1)
    scaling_factor = bits / (max_val - min_val)
    if max_val != 0:
        scaling_factor = bits / (max_val - min_val)
        act_quant = torch.clamp(torch.round(x * scaling_factor), 0, bits).type(torch.cuda.CharTensor)
        # act_quant = RNTB(x * scaling_factor)
        # act_quant = torch.clamp(act_quant, -bits, bits - 1)
        # act_dequant = act_quant / scaling_factor
    else :
        act_quant = x.type(torch.cuda.CharTensor)
        scaling_factor = 1

    return act_quant, scaling_factor

def dequantizer(x, wt_scale, act_scale):
    dequant_out = (x / (wt_scale * act_scale)).type(torch.cuda.FloatTensor)
    return dequant_out.type(torch.cuda.FloatTensor)


#Integrated Quantizer
def weight_quantization(weight, bits):
    max_val = torch.max(weight)
    min_val = torch.min(weight)

    bits = math.pow(2, bits - 1)
    scaling_factor = bits/(max_val - min_val)
    weight_quant = torch.clamp(torch.round(weight * scaling_factor), -bits, bits - 1)
    # weight_quant = RNTB(weight * scaling_factor)
    # weight_quant = torch.clamp(weight_quant, -bits, bits - 1)
    weight_dequant = weight_quant / scaling_factor
    return weight_dequant

def act_quantization(x,bits):
    max_val = torch.max(x)
    min_val = torch.min(x)

    bits = math.pow(2, bits - 1)
    scaling_factor = bits / (max_val - min_val)
    if max_val != 0:
        scaling_factor = bits / (max_val - min_val)
        act_quant = torch.clamp(torch.round(x * scaling_factor), 0, bits)
        # act_quant = RNTB(x * scaling_factor)
        # act_quant = torch.clamp(act_quant, -bits, bits - 1)
        act_dequant = act_quant / scaling_factor
    else :
        act_quant = x
        scaling_factor = 1

    return act_dequant

def pad(in_tensor, pad_h, pad_w):
    batch_num = in_tensor.shape[0]
    in_channels = in_tensor.shape[1]
    in_h = in_tensor.shape[2]
    in_w = in_tensor.shape[3]
    padded = torch.zeros([batch_num, in_channels, in_h + 2*pad_h, in_w + 2*pad_w])
    padded[:, :, pad_h:pad_h+in_h, pad_w:pad_w+in_w] = in_tensor
    return padded

def convolution(in_tensor, kernel, stride = 1, pad_size=1):
    batch_num = in_tensor.shape[0]
    in_channels = in_tensor.shape[1]
    in_h = in_tensor.shape[2]
    in_w = in_tensor.shape[3]
    out_channels = kernel.shape[0]
    assert kernel.shape[1] == in_channels
    kernel_h = kernel.shape[2]
    kernel_w = kernel.shape[3]
    
    #Decide Output shape
    out_h = int((in_h - kernel_h + 2 * pad_size) / stride + 1)
    out_w = int((in_w - kernel_w + 2 * pad_size) / stride + 1)
    if pad_size>0:
        in_tensor = pad(in_tensor, pad_size, pad_size)  
    
    #output = torch.zeros(batch_num, out_channels, out_h, out_w).type(torch.cuda.FloatTensor)
    output = torch.zeros(batch_num, out_channels, out_h, out_w).type(torch.cuda.LongTensor)
    
    #Index for stride
    h_idx = torch.zeros(out_h).type(torch.cuda.LongTensor)
    w_idx = torch.zeros(out_w).type(torch.cuda.LongTensor)
    #Apply stride
    for h in range(out_h):
        h_idx[h] = h * stride
    for w in range(out_w): 
        w_idx[w] = w * stride
    
    #Convolution computation
    for j in range(out_channels):
        for h in range(kernel_h):
            for w in range(kernel_w):
                #temp_in = in_tensor[b,:,h_idx+h,:][:,:,w_idx+w].type(torch.cuda.FloatTensor)
                temp_in = in_tensor[:, :, h_idx + h, :][:, :, :, w_idx + w].type(torch.cuda.LongTensor)
                output[:, j, :, :] += (temp_in * kernel[j , :, h, w].reshape(-1,1,1)).sum(dim=1)
                #output[b,:,:,:] += (temp_in.unsqueeze(0).repeat(out_channels,1,1,1) * kernel[:,:,h,w].reshape(out_channels,-1,1,1)).sum(dim=1)

    return output