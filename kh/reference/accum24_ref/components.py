import numpy as np
import os
import torch
from torch.nn.parameter import Parameter
import time
import torch.nn.functional as F
import math

def find_best_approximation(value, max_n=16):
    """
    Find the best approximation of a floating point number in a * 2^(-n) form.

    :param value: The floating point number to approximate.
    :param max_n: The maximum value for n to try.
    :return: Tuple (a, n, approx_value) where a and n give the best approximation.
    """
    best_approximation = None
    smallest_error = 2**-max_n
    for n in range(max_n + 1):
        # Calculate the scaled value
        scaled_value = value * (2 ** n)

        # Find the nearest integer a
        a = torch.round(scaled_value)

        # Calculate the approximate value
        approx_value = a / (2 ** n)
        
        # Calculate the error
        error = torch.abs(approx_value - value)

        # Update the best approximation if this is closer
        idx = error < smallest_error
        best_approximation = idx * approx_value

    return best_approximation


# def stochastic_round(x):

#     error_prob = x - torch.floor(x)
#     random = torch.rand_like(x)
#     rounded = torch.where(random < error_prob, torch.ceil(x), torch.floor(x))
#     return rounded

# def gradient_quantization(x):
#     bit = 8
    
#     max_val1 = torch.max(x)
#     min_val1 = torch.min(x)
#     if (torch.abs(min_val1) > max_val1):
#         clip = torch.abs(min_val1)
#     else :
#         clip = max_val1

#     clip = clip * 0.90
#     x = torch.clamp(x,-clip,clip)

#     max_val = torch.max(x)
#     min_val = torch.min(x)
#     bits = math.pow(2,bit-1)

#     s = clip
#     scaling_factor = s / bits
# #    if (max_val == 0) & (min_val == 0) :
#     if (s==0) :
#         grad_quant = x
#     else :
#         rounded_value =  x / scaling_factor
#         grad_quant = stochastic_round(rounded_value)
#         grad_quant = torch.clamp(grad_quant,-bits,bits-1).type(torch.cuda.CharTensor)

#     return grad_quant, scaling_factor.type(torch.cuda.FloatTensor)

# def accum_bit_limiter(in_data):
#     overflow_data = (in_data.abs() > 8388607).type(torch.cuda.IntTensor)
#     if overflow_data.sum() > 1:
#         keep_data = in_data.abs() <= 8388607
#         sign_data = in_data.sign()
#         overflow_data *= sign_data
#         overflow_data = overflow_data.type(torch.cuda.IntTensor) * 8388607
#         in_data *= keep_data
#         in_data += overflow_data

#         print('\n', int(overflow_data.sum()), 'FW datum are overflowed')

#     return in_data

# def bw_accum_bit_limiter(in_data):
#     overflow_data = (in_data.abs() > 8388607).type(torch.cuda.IntTensor)
#     if overflow_data.sum() > 1:
#         print('\n',overflow_data.sum(), 'BW datum are overflowed')
#         # keep_data = in_data.abs() <= 8388607
#         # sign_data = in_data.sign()
#         # overflow_data *= sign_data
#         # overflow_data = overflow_data * 8388607
#         # in_data *= keep_data
#         # in_data += overflow_data
#         in_data = torch.clamp(in_data, min=-8338607, max=8338607)
#         print('in_data == 8338607 ', (in_data == 8338607).sum())
#         print('in_data == -8338607 ', (in_data == -8338607).sum())


#     return in_data
    

def RNTB(value):
    floor_value = torch.floor(value)
    ceil_value = torch.ceil(value)
    floor_diff = value - floor_value
    sign = torch.sign(value)

    condition = (0.49 <= floor_diff) & (floor_diff <= 0.51)

    rounded_value = torch.where(condition, torch.where(sign >= 0, floor_value, ceil_value), torch.round(value))
    #num_true = torch.sum(condition).item() 
    #print("Number of True values:", num_true)

    return rounded_value

def gradient_quantizer(x):
    max_val1 = torch.max(x)
    min_val1 = torch.min(x)
    sign = torch.sign(x)
    if (torch.abs(min_val1) > max_val1):
        clip = torch.abs(min_val1)
    else :
        clip = max_val1

    bits = 8

    clip = clip * 0.90
    x = torch.clamp(x,-clip,clip)

    bits = 2 ** (bits - 1)
    s = clip

    scaling_factor = (s) / bits

    if (s==0):
        grad_quant = x.type(torch.cuda.CharTensor)
        #grad_quant = x.type(torch.cuda.ShortTensor)
        scaling_factor = torch.tensor([1.0]).type(torch.cuda.FloatTensor)
    else:
        grad_quant = RNTB(x / scaling_factor)
        grad_quant = torch.clamp(grad_quant, -bits, bits - 1).type(torch.cuda.CharTensor)
        # grad_quant[grad_quant == 0] = 1
        # grad_quant = grad_quant.abs() * sign
        #grad_quant = torch.clamp(grad_quant,-bits,bits-1).type(torch.cuda.ShortTensor)

    return grad_quant , scaling_factor.type(torch.cuda.FloatTensor)

def quantizer(x):#, bits):
    max_val = torch.max(x)
    min_val = torch.min(x)
    bit = 8
    bits = 2 ** (bit - 1)
    scaling_factor = ((max_val - min_val) / bits).type(torch.cuda.FloatTensor)
    # fixed_scaling_factor = find_best_approximation(scaling_factor, 16)

    if scaling_factor == 0:
        quant_val = x.type(torch.cuda.CharTensor)
        scaling_factor = torch.tensor([1.0]).type(torch.cuda.FloatTensor)
        # fixed_scaling_factor = torch.tensor([1.0]).type(torch.cuda.FloatTensor)
    else:
        quant_val = torch.clamp(torch.round(x / scaling_factor).type(torch.cuda.CharTensor), -bits, bits -1)
        # quant_val = torch.round(x / fixed_scaling_factor).type(torch.cuda.CharTensor)
        # quant_val = torch.round(x / scale_factor).type(torch.cuda.CharTensor)
    return quant_val, scaling_factor
    # return quant_val, fixed_scaling_factor

def pre_relu_quantizer(x):
    max_val = torch.max(x)
    min_val = torch.min(x)
    bit = 8
    bits = 2 ** (bit) -1
    scaling_factor = ((max_val - min_val) / bits).type(torch.cuda.FloatTensor)
    fixed_scaling_factor = find_best_approximation(scaling_factor, 16)

    if scaling_factor == 0:
        quant_val = x.type(torch.cuda.ShortTensor)
        #scaling_factor = torch.tensor([1.0]).type(torch.cuda.FloatTensor)
        fixed_scaling_factor = torch.tensor([1.0]).type(torch.cuda.FloatTensor)
    else:
        #quant_val = torch.round(x / scaling_factor).type(torch.cuda.ShortTensor)
        quant_val = torch.round(x / fixed_scaling_factor).type(torch.cuda.ShortTensor)

    #return quant_val, scaling_factor
    return quant_val, fixed_scaling_factor

def dequantizer(x, scale_factor_a, scale_factor_w):
    dequant_val = x * scale_factor_a * scale_factor_w
    # dequant_val = find_best_approximation(dequant_val, 16)
    return dequant_val.type(torch.cuda.FloatTensor)

def wt_dequantizer(x, scale_factor_w):
    dequant_val = x * scale_factor_w
    # dequant_val = find_best_approximation(dequant_val, 16)
    return dequant_val.type(torch.cuda.FloatTensor)

def error_find_dequant(origin_x, quant_x, scale_factor):
    dequant_val = quant_x * scale_factor
    error = origin_x - dequant_val
    print('\nerror :', error.sum())

class upsample:
    def __init__(self, mode):
        assert mode == 'proto' or mode == 'fpn'
        
        if mode == 'proto': 
            self.switch = True
        elif mode == 'fpn':
            self.switch = False
        

    def forward(self, inputs):    
        output = torch.zeros(inputs.shape[0], inputs.shape[1], inputs.shape[2] * 2, inputs.shape[3] * 2).type(torch.cuda.FloatTensor)
        x_idx = torch.arange(inputs.shape[2]) * 2

        for r in range(2):
            for c in range(2):
                for i in range(inputs.shape[3]):
                    output[:,:,x_idx+r,i*2+c] = inputs[:,:,:,i]
        
        if self.switch==False:
            output = output[:,:,:-1,:-1]
        
        return output
    
    def monitor(self):
            print('Downsample Loss input(abs) :', self.loss_input.abs().sum())
            print('Downsample Loss output(abs) :', self.loss_output.abs().sum())
    
    def backward(self, loss_input):
        self.loss_input = loss_input
        
        if self.switch==True:
            loss_output = torch.zeros(loss_input.shape[0], loss_input.shape[1], loss_input.shape[2]//2, loss_input.shape[3]//2).type(torch.cuda.FloatTensor)
        else:
            loss_output = torch.zeros(loss_input.shape[0], loss_input.shape[1], (loss_input.shape[2]+1)//2, (loss_input.shape[3]+1)//2).type(torch.cuda.FloatTensor)

        x_idx = torch.arange(loss_input.shape[3]//2) * 2
        
        
        if self.switch==True:       
            for r in range(2):
                for c in range(2):
                    for i in range(loss_input.shape[3]//2):
                        loss_output[:,:,:,i] += loss_input[:,:,x_idx + r, x_idx[i] + c]
        
        else:
            for r in range(2):
                for c in range(2):
                    for i in range(loss_input.shape[3]//2):
                        loss_output[:,:,:-1,i] += loss_input[:,:,x_idx + r, x_idx[i] + c]                    
            for c in range(2):
                loss_output[:,:,-1,:-1] += loss_input[:,:,-1,x_idx+c]
            for r in range(2):
                loss_output[:,:,:-1,-1] += loss_input[:,:,x_idx+r,-1]
            loss_output[:,:,-1,-1] = loss_input[:,:,-1,-1]

        self.loss_output = loss_output
        
        return loss_output

        



class tanh:
    def __init__(self):
        self.out_tensor = []

    def forward(self, in_tensor):
        self.out_tensor.append(torch.tanh(in_tensor))
        return self.out_tensor[-1]
        
    def backward(self, out_diff_tensor, num):
        assert self.out_tensor[num].shape == out_diff_tensor.shape
        self.error = out_diff_tensor.clone()*(1 - self.out_tensor[num]**2)
        return self.error
    
    def refresh(self):
        self.out_tensor = []

class softmax:
    def forward(self, in_tensor, dim):
        exp_a = torch.exp(in_tensor)
        sum_exp_a = torch.sum(exp_a, dim=dim).reshape(-1,exp_a.shape[1],1)
        sum_exp_a = sum_exp_a.repeat_interleave(exp_a.shape[-1], dim=dim)
        y = exp_a / sum_exp_a
        return y



class fc_sigmoid:

    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_param()

    def init_param(self):
        self.kernel = torch.cuda.FloatTensor(self.out_channels, self.in_channels).uniform_(-torch.sqrt(6.0/torch.tensor(self.out_channels + self.in_channels, dtype=torch.float32)), torch.sqrt(6.0/torch.tensor(self.in_channels + self.out_channels, dtype=torch.float32)))
        self.bias = torch.zeros([self.out_channels]).type(torch.cuda.FloatTensor)

    def forward(self, in_tensor):
        self.shape = in_tensor.shape
        self.in_tensor = in_tensor.reshape(in_tensor.shape[0], -1).clone()
        assert self.in_tensor.shape[1] == self.kernel.shape[1]
        self.out_tensor = torch.matmul(self.in_tensor, self.kernel.T) + self.bias.T
        self.out_tensor = 1.0 / (1.0 + torch.exp(-self.out_tensor))
        return self.out_tensor

    def backward(self, out_diff_tensor, lr):
        assert out_diff_tensor.shape == self.out_tensor.shape
        nonlinear_diff = self.out_tensor * (1 - self.out_tensor) * out_diff_tensor
        kernel_diff = torch.matmul(nonlinear_diff.T, self.in_tensor).squeeze()
        bias_diff = torch.sum(nonlinear_diff, dim=0).reshape(self.bias.shape)
        self.in_diff_tensor = torch.matmul(nonlinear_diff, self.kernel).reshape(self.shape)
        self.kernel -= lr * kernel_diff
        self.bias -= lr * bias_diff

    def save(self, path):
        if os.path.exists(path) == False:
            os.mkdir(path)

        np.save(os.path.join(path, "fc_weight.npy"), self.kernel)
        np.save(os.path.join(path, "fc_bias.npy"), self.bias)

    def load(self, path):
        assert os.path.exists(path)

        self.kernel = np.load(os.path.join(path, "fc_weight.npy"))
        self.bias = np.load(os.path.join(path, "fc_bias.npy"))



class conv_layer:
    def __init__(self, in_channels, out_channels, kernel_h, kernel_w, same = True, stride = 1, shift = True, pad = 1, pred = False, quant=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.same = same
        self.stride = stride
        self.shift = shift
        self.pad_size = pad
        self.weight = Parameter(torch.zeros(self.out_channels, self.in_channels, self.kernel_h, self.kernel_w).type(torch.cuda.FloatTensor), requires_grad=False)
        self.grad_weight = self.weight.clone().detach()
        self.bias = Parameter(torch.zeros(self.out_channels).type(torch.cuda.FloatTensor), requires_grad=False)
        self.in_tensor = []
        self.in_scale = []
        self.out_tensor = []
        self.conv_dq_out = []
        self.quant = quant
        self.mean_cnt = [1]
        self.act_scale_mean = [0.0]
        self.pred_cnt = 0
        if shift:
            self.grad_bias = Parameter(torch.zeros(self.out_channels), requires_grad=False).type(torch.cuda.FloatTensor) 
        if pred:
            self.bias_diff = torch.zeros(self.grad_bias.shape)
            self.kernel_diff = torch.zeros(self.grad_weight.shape)


    @staticmethod
    def pad(in_tensor, pad_h, pad_w):
        batch_num = in_tensor.shape[0]
        in_channels = in_tensor.shape[1]
        in_h = in_tensor.shape[2]
        in_w = in_tensor.shape[3]
        padded = torch.zeros([batch_num, in_channels, in_h + 2*pad_h, in_w + 2*pad_w])
        padded[:, :, pad_h:pad_h+in_h, pad_w:pad_w+in_w] = in_tensor
        return padded
    
    def load(self, conv_num): 
        if self.quant:
            self.weight = conv_num[0][0].cuda()
            # self.wt_scale = find_best_approximation(conv_num[0][1].cuda(), 16)
            self.wt_scale = conv_num[0][1].cuda()
        else:
            self.weight = conv_num[0].cuda()
        
        if self.shift:
            # self.bias = find_best_approximation(conv_num[1].cuda(), 16)
            self.bias = conv_num[1].cuda()
    
    @staticmethod
    #Small kernel convolution
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
        if pad_size > 0:
            in_tensor = conv_layer.pad(in_tensor, pad_size, pad_size)  
        
        #output = torch.zeros(batch_num, out_channels, out_h, out_w).type(torch.cuda.FloatTensor)
        output = torch.zeros(batch_num, out_channels, out_h, out_w).type(torch.cuda.IntTensor)
        
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
                    temp_in = in_tensor[:, :, h_idx + h, :][:, :, :, w_idx + w].type(torch.cuda.IntTensor)
                    output[:, j, :, :] += (temp_in * kernel[j, :, h, w].reshape(-1,1,1)).sum(dim=1)
                    #output[b,:,:,:] += (temp_in.unsqueeze(0).repeat(out_channels,1,1,1) * kernel[:,:,h,w].reshape(out_channels,-1,1,1)).sum(dim=1)

        return output
    
    def convolution_first(in_tensor, kernel, stride = 1, pad_size=1):
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
            in_tensor = conv_layer.pad(in_tensor, pad_size, pad_size)  
        
        #output = torch.zeros(batch_num, out_channels, out_h, out_w).type(torch.cuda.FloatTensor)
        output = torch.zeros(batch_num, out_channels, out_h, out_w).type(torch.cuda.FloatTensor)
        
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
                    temp_in = in_tensor[:, :, h_idx + h, :][:, :, :, w_idx + w ].type(torch.cuda.FloatTensor)
                    output[:, j, :, :] += (temp_in * kernel[j, :, h, w].reshape(-1,1,1)).sum(dim=1)
                    #output[b,:,:,:] += (temp_in.unsqueeze(0).repeat(out_channels,1,1,1) * kernel[:,:,h,w].reshape(out_channels,-1,1,1)).sum(dim=1)

        return output
        
    #Large kernel convolution (For backward)
    def w_grad_convolution(in_tensor, kernel, stride = 1, pad_size=0):
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
        if pad_size > 0:
            in_tensor = conv_layer.pad(in_tensor, pad_size, pad_size)

        output = torch.zeros(batch_num, out_channels, out_h, out_w).type(torch.cuda.IntTensor)
        
        #Convolution computation
        for b in range(batch_num):
            for r in range(out_h):
                for c in range(out_w): 
                    temp_in = in_tensor[b].type(torch.cuda.IntTensor)
                    output[b, :, r, c] += (kernel * temp_in[:, r : r + kernel_h, c : c + kernel_w]).reshape(kernel.shape[0],-1).sum(dim=1)

        return output

    
    def forward(self, in_tensor):
        #Save input & output for backward process
        self.in_tensor = []
        self.in_scale = []
        self.out_tensor = []
        self.conv_dq_out = []
        quant_act, act_scale = quantizer(in_tensor)
        # error_find_dequant(in_tensor, quant_act, act_scale)
        # if self.mean_cnt[-1] == 1:
        #     self.act_scale_mean[-1] = act_scale
        # else:
        #     self.act_scale_mean[-1] = (self.act_scale_mean[-1] * self.mean_cnt[-1]  + act_scale) / (self.mean_cnt[-1] + 1)
        # self.mean_cnt[-1] += 1

        self.in_tensor.append(quant_act.clone().detach()) 
        self.in_scale.append(act_scale)
        quant_out = conv_layer.convolution(self.in_tensor[-1].type(torch.cuda.IntTensor), self.weight.type(torch.cuda.IntTensor), self.stride, pad_size = self.pad_size).clone().detach()
        #self.out_tensor.append(conv_layer.convolution(self.in_tensor[-1], quant_wt, self.stride, pad_size = self.pad_size).clone().detach())
        quant_out = torch.clamp(quant_out, min = -8388607, max = 8388607)
        dequant_out = dequantizer(quant_out, act_scale, self.wt_scale)
        self.conv_dq_out = dequantizer(quant_out, act_scale, self.wt_scale)
        self.out_tensor.append(dequant_out)

        self.quant_out = quant_out

        if self.shift:
            #self.conv_dq_out.append(dequant_out.clone())
            self.out_tensor[-1] += self.bias.reshape(1, self.out_channels, 1, 1)

        return self.out_tensor[-1]
    
    def pre_relu_forward(self, in_tensor):
        #Save input & output for backward process
        self.in_tensor = []
        self.in_scale = []
        self.out_tensor = []
        quant_act, act_scale = pre_relu_quantizer(in_tensor)

        self.in_tensor.append(quant_act.clone().detach()) 
        self.in_scale.append(act_scale)
        quant_out = conv_layer.convolution(self.in_tensor[-1].type(torch.cuda.IntTensor), self.weight.type(torch.cuda.IntTensor), self.stride, pad_size = self.pad_size).clone().detach()
        #self.out_tensor.append(conv_layer.convolution(self.in_tensor[-1], quant_wt, self.stride, pad_size = self.pad_size).clone().detach())
        dequant_out = dequantizer(quant_out, act_scale, self.wt_scale)
        self.out_tensor.append(dequant_out)

        self.quant_out = quant_out

        if self.shift:
            self.out_tensor[-1] += self.bias.reshape(1, self.out_channels, 1, 1)

        return self.out_tensor[-1]
    
    def forward_pred(self, in_tensor):
        quant_act, act_scale = quantizer(in_tensor)
        # error_find_dequant(in_tensor, quant_act, act_scale)

        self.in_tensor.append(quant_act.clone().detach())
        self.in_scale.append(act_scale)

        # if self.mean_cnt[self.pred_cnt] == 1:
        #     self.act_scale_mean[self.pred_cnt] = act_scale
        #     if self.pred_cnt == 0 and len(self.mean_cnt) == 1:
        #         for _ in range(4):
        #             self.mean_cnt.append(1)
        #             self.act_scale_mean.append(0.0)
        # else:
        #     self.act_scale_mean[self.pred_cnt] = (self.act_scale_mean[self.pred_cnt] * self.mean_cnt[self.pred_cnt]  + act_scale) / (self.mean_cnt[self.pred_cnt] + 1)
        # self.mean_cnt[self.pred_cnt] += 1
        # self.pred_cnt += 1
        # if self.pred_cnt == 5:
        #     self.pred_cnt = 0

        quant_out = conv_layer.convolution(self.in_tensor[-1].type(torch.cuda.IntTensor), self.weight.type(torch.cuda.IntTensor), self.stride, pad_size = self.pad_size)
        quant_out = torch.clamp(quant_out, min = -8388607, max = 8388607)
        dequant_out = dequantizer(quant_out, act_scale, self.wt_scale)
        self.conv_dq_out.append(dequantizer(quant_out, act_scale, self.wt_scale))
        self.out_tensor.append(dequant_out)

        self.quant_out = quant_out

        if self.shift:
            #self.conv_dq_out.append(dequant_out)
            self.out_tensor[-1] += self.bias.reshape(1, self.out_channels, 1, 1)

        return self.out_tensor[-1]
    
    def pre_relu_forward_pred(self, in_tensor):
        quant_act, act_scale = pre_relu_quantizer(in_tensor)

        self.in_tensor.append(quant_act.clone().detach())
        self.in_scale.append(act_scale)
        quant_out = conv_layer.convolution(self.in_tensor[-1].type(torch.cuda.IntTensor), self.weight.type(torch.cuda.IntTensor), self.stride, pad_size = self.pad_size)
        
        dequant_out = dequantizer(quant_out, act_scale, self.wt_scale)
        self.out_tensor.append(dequant_out)

        self.quant_out = quant_out

        if self.shift:
            self.out_tensor[-1] += self.bias.reshape(1, self.out_channels, 1, 1)

        return self.out_tensor[-1]
    
    def backward(self, out_diff_tensor, lr, weight_decay, momentum):
        # assert out_diff_tensor.shape == self.out_tensor[0].shape

        self.loss_input = out_diff_tensor
        if self.shift:
            bias_diff = torch.sum(out_diff_tensor, dim = (0,2,3)).reshape(self.bias.shape)
            bias_diff = bias_diff + weight_decay * self.bias
            self.grad_bias = momentum * self.grad_bias + bias_diff
            self.bias = self.bias - lr * self.grad_bias
        
        batch_num = out_diff_tensor.shape[0]
        out_channels = out_diff_tensor.shape[1]
        out_h = out_diff_tensor.shape[2]
        out_w = out_diff_tensor.shape[3]
        extend_out = torch.zeros([batch_num, out_channels, out_h, out_w, self.stride * self.stride]).type(torch.cuda.FloatTensor)
        extend_out[:, :, :, :, 0] = out_diff_tensor.type(torch.cuda.FloatTensor).clone().detach()
        extend_out = extend_out.reshape(batch_num, out_channels, out_h, out_w, self.stride, self.stride)
        extend_out = extend_out.permute((0,1,2,4,3,5)).reshape(batch_num, out_channels, out_h * self.stride, out_w * self.stride)
        
        if extend_out.shape[3] > self.in_tensor[0].shape[3]:
            extend_out = extend_out[:,:,:-1,:-1]

        #Loss quantization
        quant_loss, loss_scale = gradient_quantizer(extend_out)

        #Weight Gradient Convolution operation
        kernel_diff = conv_layer.w_grad_convolution(self.in_tensor[-1].permute(1,0,2,3).type(torch.cuda.IntTensor), quant_loss.permute(1,0,2,3).type(torch.cuda.IntTensor), pad_size = self.pad_size)
        kernel_diff = torch.clamp(kernel_diff, min = -8388607, max = 8388607)
        kernel_diff = kernel_diff.permute(1,0,2,3)
        
        #Weight Gradient Dequantization
        dequant_wt_grad = dequantizer(kernel_diff, self.in_scale[-1], loss_scale)


        #Loss backpropagation Process
        #Weight 180 rotation
        kernel_trans = torch.rot90(self.weight.reshape(self.out_channels, self.in_channels, self.kernel_h, self.kernel_w), 2, [2, 3]).clone().detach()
        #Loss backprop calculation
        int_loss = conv_layer.convolution(quant_loss.type(torch.cuda.IntTensor), kernel_trans.permute(1,0,2,3).type(torch.cuda.IntTensor), stride = 1, pad_size = self.kernel_h - 1)
        int_loss = torch.clamp(int_loss, min = -8388607, max = 8388607)
        #Updated Loss Dequantization
        dequant_loss = dequantizer(int_loss, loss_scale, self.wt_scale)


        if self.same:
            pad_h = int((self.kernel_h-1)/2)
            pad_w = int((self.kernel_w-1)/2)
            if pad_h == 0 and pad_w != 0:
                dequant_loss = dequant_loss[:, :, :, pad_w:-pad_w]
            elif pad_h !=0 and pad_w == 0:
                dequant_loss = dequant_loss[:, :, pad_h:-pad_h, :]
            elif pad_h != 0 and pad_w != 0:
                dequant_loss = dequant_loss[:, :, pad_h:-pad_h, pad_w:-pad_w]

        #Dequant weight for update
        dequant_weight = wt_dequantizer(self.weight, self.wt_scale)

        dequant_wt_grad = dequant_wt_grad + weight_decay * dequant_weight
        self.grad_weight = momentum * self.grad_weight + dequant_wt_grad
        dequant_weight = dequant_weight - lr * self.grad_weight.clone().detach()

        #Quant weight
        self.weight, self.wt_scale = quantizer(dequant_weight)

        assert dequant_loss.shape == self.in_tensor[0].shape
        #Memory free
        self.in_tensor = []
        self.out_tensor = []
        self.conv_dq_out = []

        return dequant_loss

    
    def backward_pred(self, out_diff_tensor, num):
        assert out_diff_tensor.shape == self.out_tensor[num].shape

        self.loss_input = out_diff_tensor
        if self.shift:
            bias_diff = torch.sum(out_diff_tensor, dim = (0,2,3)).reshape(self.bias.shape)
            self.bias_diff = self.bias_diff.clone().detach() + bias_diff
        
        batch_num = out_diff_tensor.shape[0]
        out_channels = out_diff_tensor.shape[1]
        out_h = out_diff_tensor.shape[2]
        out_w = out_diff_tensor.shape[3]
        extend_out = torch.zeros([batch_num, out_channels, out_h, out_w, self.stride * self.stride])
        extend_out[:, :, :, :, 0] = out_diff_tensor.clone().detach()
        extend_out = extend_out.reshape(batch_num, out_channels, out_h, out_w, self.stride, self.stride)
        extend_out = extend_out.permute((0,1,2,4,3,5)).reshape(batch_num, out_channels, out_h*self.stride, out_w*self.stride)

        #Loss quantization
        quant_loss, loss_scale = gradient_quantizer(extend_out)
        # quant_loss, loss_scale = quantizer(extend_out)

        #Weight Gradient Convolution Operation
        kernel_diff = conv_layer.w_grad_convolution(self.in_tensor[num].permute(1,0,2,3).type(torch.cuda.IntTensor), quant_loss.permute(1,0,2,3).type(torch.cuda.IntTensor))
        kernel_diff = torch.clamp(kernel_diff, min = -8388607, max = 8388607)
        kernel_diff = kernel_diff.permute(1,0,2,3)
        #Weight Gradient Dequantization
        dequant_wt_grad = dequantizer(kernel_diff, loss_scale, self.in_scale[num])
        self.kernel_diff = self.kernel_diff.clone().detach() + dequant_wt_grad
        
        #padded = conv_layer.pad(extend_out, self.kernel_h-1, self.kernel_w-1)
        kernel_trans = torch.rot90(self.weight.reshape(self.out_channels, self.in_channels, self.kernel_h, self.kernel_w), 2, [2,3])
        #Loss backprop calculation
        int_loss = conv_layer.convolution(quant_loss, kernel_trans.permute(1,0,2,3), pad_size = self.kernel_h-1)
        int_loss = torch.clamp(int_loss, min = -8388607, max = 8388607)
        #Dequantization Loss
        dequant_loss = dequantizer(int_loss, loss_scale, self.wt_scale)


        if self.same:
            pad_h = int((self.kernel_h-1)/2)
            pad_w = int((self.kernel_w-1)/2)
            if pad_h == 0 and pad_w != 0:
                dequant_loss = dequant_loss[:, :, :, pad_w:-pad_w]
            elif pad_h !=0 and pad_w == 0:
                dequant_loss = dequant_loss[:, :, pad_h:-pad_h, :]
            elif pad_h != 0 and pad_w != 0:
                dequant_loss = dequant_loss[:, :, pad_h:-pad_h, pad_w:-pad_w]
        
        assert dequant_loss.shape == self.in_tensor[num].shape
        
        return dequant_loss
    
    def step(self, lr, weight_decay, momentum):
        dequant_wt = wt_dequantizer(self.weight, self.wt_scale)
        self.kernel_diff = self.kernel_diff + weight_decay * dequant_wt
        self.grad_weight = momentum * self.grad_weight.clone().detach() + self.kernel_diff.clone().detach()
        dequant_wt = dequant_wt - lr * self.grad_weight.clone().detach()
        
        #Quantize updated weight
        self.weight, self.wt_scale = quantizer(dequant_wt)

        if self.shift:
            self.bias_diff = self.bias_diff.clone().detach() + weight_decay * self.bias.clone().detach()
            self.grad_bias = momentum * self.grad_bias.clone().detach() + self.bias_diff.clone().detach()
            self.bias = self.bias - lr * self.grad_bias.clone().detach()
            self.bias = find_best_approximation(self.bias, 16)
        #Memory free
        self.in_tensor = []
        self.in_scale = []
        self.out_tensor = []
        self.conv_dq_out = []
        
        #Initialization
        self.bias_diff.fill_(0)
        self.kernel_diff.fill_(0)
    
    def refresh(self):
        self.in_tensor = []
        self.in_scale = []
        self.out_tensor = []
        self.conv_dq_out = []

    def monitor(self):
        print('Loss input(abs) :', self.loss_input.shape, self.loss_input.abs().sum())
        print('grad_weight(abs) :', self.grad_weight.shape, self.grad_weight.abs().sum(), self.grad_weight[0,:5,:,:].reshape(5,-1).abs().sum(dim=1))
        print('Loss output(abs) :', self.error.shape, self.error.abs().sum())

class max_pooling:
    
    def __init__(self, kernel_h, kernel_w, stride, same=False):
        assert stride > 1
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.same = same
        self.stride = stride

    @staticmethod
    def pad(in_tensor, pad_h, pad_w):
        batch_num = in_tensor.shape[0]
        in_channels = in_tensor.shape[1]
        in_h = in_tensor.shape[2]
        in_w = in_tensor.shape[3]
        #padded = np.zeros([batch_num, in_channels, in_h + 2*pad_h, in_w + 2*pad_w])
        padded = torch.zeros([batch_num, in_channels, in_h + 2*pad_h, in_w + 2*pad_w])
        padded[:, :, pad_h:pad_h+in_h, pad_w:pad_w+in_w] = in_tensor
        return padded

    def forward(self, in_tensor):
        if self.same:
            in_tensor = max_pooling.pad(in_tensor, int((self.kernel_h-1)/2), int((self.kernel_w-1)/2))
        self.shape = in_tensor.shape

        batch_num = in_tensor.shape[0]
        in_channels = in_tensor.shape[1]
        in_h = in_tensor.shape[2]
        in_w = in_tensor.shape[3]
        out_h = int((in_h - self.kernel_h) / self.stride) + 1
        out_w = int((in_w - self.kernel_w) / self.stride) + 1

        #out_tensor = np.zeros([batch_num, in_channels, out_h, out_w])
        out_tensor = torch.zeros([batch_num, in_channels, out_h, out_w])
        #self.maxindex = np.zeros([batch_num, in_channels, out_h, out_w], dtype = torch.int32)
        self.maxindex = torch.zeros([batch_num, in_channels, out_h, out_w], dtype = torch.int32)
        for i in range(out_h):
            for j in range(out_w):
                part = in_tensor[:, :, i*self.stride:i*self.stride+self.kernel_h, j*self.stride:j*self.stride+self.kernel_w].reshape(batch_num, in_channels, -1)
                #out_tensor[:, :, i, j] = np.max(part, axis = -1)
                out_tensor[:, :, i, j], _ = torch.max(part, dim = -1)
                #self.maxindex[:, :, i, j] = np.argmax(part, axis = -1)
                self.maxindex[:, :, i, j] = torch.argmax(part, dim = -1)
        self.out_tensor = out_tensor
        return self.out_tensor

    def backward(self, out_diff_tensor):
        assert out_diff_tensor.shape == self.out_tensor.shape
        batch_num = out_diff_tensor.shape[0]
        in_channels = out_diff_tensor.shape[1]
        out_h = out_diff_tensor.shape[2]
        out_w = out_diff_tensor.shape[3]
        in_h = self.shape[2]
        in_w = self.shape[3]

        out_diff_tensor = out_diff_tensor.reshape(batch_num*in_channels, out_h, out_w)
        self.maxindex = self.maxindex.reshape(batch_num*in_channels, out_h, out_w)
        
        #self.in_diff_tensor = np.zeros([batch_num*in_channels, in_h, in_w])
        self.in_diff_tensor = torch.zeros([batch_num*in_channels, in_h, in_w])
        #h_index = (self.maxindex/self.kernel_h).astype(np.int32)
        h_index = (self.maxindex/self.kernel_h).type(torch.cuda.LongTensor)
        w_index = (self.maxindex - h_index * self.kernel_h).type(torch.cuda.LongTensor)
        for i in range(out_h):
            for j in range(out_w):
                self.in_diff_tensor[range(batch_num * in_channels), i*self.stride+h_index[:,i,j], j*self.stride+w_index[:,i,j]] += out_diff_tensor[:,i,j]
        self.in_diff_tensor = self.in_diff_tensor.reshape(batch_num, in_channels, in_h, in_w)

        if self.same:
            pad_h = int((self.kernel_h - 1) / 2)
            pad_w = int((self.kernel_w - 1) / 2)
            self.in_diff_tensor = self.in_diff_tensor[:, :, pad_h:-pad_h, pad_w:-pad_w]
        
        return self.in_diff_tensor



class global_average_pooling:
    
    def forward(self, in_tensor):
        self.shape = in_tensor.shape
        out_tensor = in_tensor.reshape(in_tensor.shape[0], in_tensor.shape[1], -1).mean(axis = -1)
        return out_tensor.reshape(in_tensor.shape[0], in_tensor.shape[1], 1, 1)

    def backward(self, out_diff_tensor, lr=0):
        batch_num = self.shape[0]
        in_channels = self.shape[1]
        in_h = self.shape[2]
        in_w = self.shape[3]
        assert out_diff_tensor.shape == (batch_num, in_channels, 1, 1)

        #in_diff_tensor = np.zeros(list(self.shape))
        in_diff_tensor = torch.zeros(list(self.shape))
        in_diff_tensor += out_diff_tensor / (in_h * in_w)
        
        self.in_diff_tensor = in_diff_tensor



class relu:
    def __init__(self):
        self.in_tensor = []
        self.out_tensor = []
        
    def forward(self, in_tensor):
        self.in_tensor = []
        self.out_tensor = []

        self.in_tensor.append(in_tensor.clone()) 
        self.out_tensor.append(in_tensor.clone()) 
        self.out_tensor[-1][self.in_tensor[-1] < 0.0] = 0.0
        return self.out_tensor[-1]
    
    def forward_pred(self, in_tensor):
        self.in_tensor.append(in_tensor.clone()) 
        self.out_tensor.append(in_tensor.clone()) 
        self.out_tensor[-1][self.in_tensor[-1] < 0.0] = 0.0
        return self.out_tensor[-1]

    def backward(self, out_diff_tensor):
        assert self.out_tensor[0].shape == out_diff_tensor.shape
        self.loss_input = out_diff_tensor
        self.error = out_diff_tensor.clone().detach()
        self.error[self.in_tensor[0] < 0.0] = 0.0
        #self.in_tensor = []
        #self.out_tensor = []
        return self.error
    
    def backward_pred(self, out_diff_tensor, num):
        assert self.out_tensor[num].shape == out_diff_tensor.shape
        self.error = out_diff_tensor.clone().detach()
        self.error[self.in_tensor[num] < 0.0] = 0.0
        return self.error
    
    def refresh(self):
        self.in_tensor = []
        self.out_tensor = []

    def monitor(self):
        print('ReLU Loss input(abs) :', self.loss_input.abs().sum())
        print('ReLU Loss output(abs) :', self.error.abs().sum())
        self.in_tensor = []
        self.out_tensor = []




class bn_layer:

    def __init__(self, neural_num, moving_rate = 0.1):
        self.gamma = Parameter(torch.zeros(neural_num), requires_grad=False).type(torch.cuda.FloatTensor)
        self.bias = Parameter(torch.zeros([neural_num]), requires_grad=False).type(torch.cuda.FloatTensor)
        self.moving_avg = Parameter(torch.zeros([neural_num]), requires_grad=False)
        self.moving_avg = self.moving_avg.cuda()
        self.moving_var = Parameter(torch.ones([neural_num]), requires_grad=False)
        self.moving_var = self.moving_var.cuda()
        self.neural_num = neural_num
        self.moving_rate = moving_rate
        self.is_train = False
        self.epsilon = 1e-5
        self.frozen = False
        self.grad_gamma = Parameter(torch.zeros(neural_num), requires_grad=False).type(torch.cuda.FloatTensor)
        self.grad_bias = self.bias

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False
    
    def bn_freeze(self):
        self.is_train = False
        self.frozen = True
        self.freeze_mean = self.moving_avg.reshape(1,-1,1,1).clone().detach()
        self.freeze_var = self.moving_var.clone().detach() + self.epsilon
        self.sqrt_var = torch.sqrt(self.moving_var.clone().detach() + self.epsilon)

    def forward(self, in_tensor):
        assert in_tensor.shape[1] == self.neural_num

        self.in_tensor = in_tensor.clone()

        if self.is_train:
            mean = in_tensor.mean(axis=(0,2,3))
            var = in_tensor.var(axis=(0,2,3))
            self.moving_avg = mean * self.moving_rate.clone().detach() + (1 - self.moving_rate.clone().detach()) * self.moving_avg.clone().detach()
            self.moving_var = var * self.moving_rate.clone().detach() + (1 - self.moving_rate.clone().detach()) * self.moving_var.clone().detach()
            self.var = var
            self.mean = mean
        else:
            mean = self.moving_avg.clone().detach()
            var = self.moving_var.clone().detach()
            self.mean = mean.clone().detach()
            self.var = var.clone().detach()

        #print(self.normalized.type())
        if self.frozen == True:
            self.normalized = (in_tensor - self.freeze_mean) / self.sqrt_var.reshape(1,-1,1,1)
        else:
            self.normalized = (in_tensor - mean.reshape(1,-1,1,1)) / torch.sqrt(var.reshape(1,-1,1,1)+self.epsilon)
        out_tensor = self.gamma.reshape(1,-1,1,1) * self.normalized + self.bias.reshape(1,-1,1,1)
        self.out_tensor = out_tensor

        return out_tensor 

    def backward(self, out_diff_tensor, lr, weight_decay, momentum):
        assert out_diff_tensor.shape == self.in_tensor.shape
        self.loss_input = out_diff_tensor

        m = self.in_tensor.shape[0] * self.in_tensor.shape[2] * self.in_tensor.shape[3]

        normalized_diff = self.gamma.reshape(1,-1,1,1).clone().detach() * out_diff_tensor.clone().detach()
        # if self.frozen:
        #     var_diff = -0.5 * torch.sum(normalized_diff * self.normalized, dim=(0,2,3)) / self.freeze_var
        #     mean_diff = -1.0 * torch.sum(normalized_diff, dim=(0,2,3)) / self.sqrt_var
        #     in_diff_tensor1 = normalized_diff / self.sqrt_var.reshape(1,-1,1,1)
        #     in_diff_tensor2 = var_diff.reshape(1,-1,1,1) * (self.in_tensor.clone().detach() - self.freeze_mean) * 2 / m
        #     in_diff_tensor3 = mean_diff.reshape(1,-1,1,1) / m
        #     self.in_diff_tensor = in_diff_tensor1 + in_diff_tensor2 + in_diff_tensor3
        # else:
        #var_diff = -0.5 * torch.sum(normalized_diff*self.normalized, dim=(0,2,3)) / (self.var.clone().detach() + self.epsilon)
        var_diff = -0.5 * normalized_diff*self.normalized / (self.var.reshape(1,-1,1,1).clone().detach() + self.epsilon)
        #mean_diff = -1.0 * torch.sum(normalized_diff, dim=(0,2,3)) / torch.sqrt(self.var.clone().detach() + self.epsilon)
        mean_diff = -1.0 * normalized_diff / torch.sqrt(self.var.reshape(1,-1,1,1).clone().detach() + self.epsilon)
        in_diff_tensor1 = normalized_diff / torch.sqrt(self.var.reshape(1,-1,1,1).clone().detach() + self.epsilon)
        #in_diff_tensor2 = var_diff.reshape(1,-1,1,1) * (self.in_tensor.clone().detach() - self.mean.reshape(1,-1,1,1)) * 2 / m
        #in_diff_tensor3 = mean_diff.reshape(1,-1,1,1) / m

        in_diff_tensor2 = var_diff * (self.in_tensor.clone().detach() - self.mean.reshape(1,-1,1,1)) * 2 / m
        in_diff_tensor3 = mean_diff / m
        self.in_diff_tensor = in_diff_tensor1 + in_diff_tensor2 + in_diff_tensor3

        gamma_diff = torch.sum(self.normalized * out_diff_tensor, dim=(0,2,3)).clone().detach()
        gamma_diff = gamma_diff + weight_decay * self.gamma
        self.grad_gamma = momentum * self.grad_gamma.clone().detach() + gamma_diff.clone().detach()
        self.gamma = self.gamma.clone().detach() - lr * self.grad_gamma.clone().detach()

        bias_diff = torch.sum(out_diff_tensor, dim=(0,2,3)).clone().detach()
        bias_diff = bias_diff + weight_decay * self.bias.clone().detach()
        self.grad_bias = momentum * self.grad_bias.clone().detach() + bias_diff
        self.bias = self.bias - lr * self.grad_bias .clone().detach()
        
        return self.in_diff_tensor.clone().detach()
        
    def monitor(self):
        print('BN Loss Input / (abs) :', self.loss_input.sum(), self.loss_input.abs().sum())
        print('BN Loss Output / (abs) :', self.in_diff_tensor.shape, self.in_diff_tensor.sum(), self.in_diff_tensor.abs().sum(), self.in_diff_tensor[0,:5,:,:].reshape(5,-1).abs().sum(dim=1)) 
        #print('BN Loss Input :', self.loss_input.shape, self.loss_input.sum())
        #print('BN Loss Output :', self.in_diff_tensor.shape, self.in_diff_tensor.sum()) 
        
        
    def load(self, bn_num):

        self.gamma = find_best_approximation(bn_num[0].cuda(), 16)
        self.bias = find_best_approximation(bn_num[1].cuda(), 16)
        self.moving_avg = find_best_approximation(bn_num[2].cuda(), 16)
        self.moving_var = find_best_approximation(bn_num[3].cuda(), 16)
        self.std = 1 / torch.sqrt(self.moving_var.reshape(1,-1,1,1) + self.epsilon)

