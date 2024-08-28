import torch, torchvision
import torch.nn as nn
import argparse 
import os
import numpy as np



parser = argparse.ArgumentParser(description='Prune Yolact')

parser.add_argument('--path', default='weights/', help='Directory for load weight.')
parser.add_argument('--save', default='weights/quant/', help='Directory for save weight.')

args = parser.parse_args()

torch.cuda.current_device()


use_jit = torch.cuda.device_count() <= 1
if not use_jit:
    print('Multiple GPUs detected! Turning off JIT.')

ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn

path = args.path
print('path :', path)

def error_find_dequant(origin_x, quant_x, scale_factor):
    dequant_val = quant_x * scale_factor
    error = origin_x - dequant_val
    print('\nerror :', error.sum())

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
    if (torch.abs(min_val1) > max_val1):
        clip = torch.abs(min_val1)
    else :
        clip = max_val1

    bits = 8

    clip = clip * 0.90
    x = torch.clamp(x,-clip,clip)

    bits = 2 ** (bits - 1)
    s = clip

    scaling_factor = bits/(s)

    #rounded_value = x * scaling_factor
    #grad_quant = stochastic_round(rounded_value)
    #grad_quant = torch.clamp(torch.round((x)*scaling_factor),-nearest_bits,nearest_bits-1)
    grad_quant = RNTB(x * scaling_factor)
    grad_quant = torch.clamp(grad_quant,-bits,bits-1).type(torch.cuda.CharTensor)
    #grad_dequant = (grad_quant / scaling_factor ).type(torch.cuda.CharTensor)

    return grad_quant , 1 / scaling_factor.type(torch.cuda.FloatTensor)

def quantizer(x):#, bits):
    max_val = torch.max(x)
    min_val = torch.min(x)
    bit = 8
    bits = 2 ** (bit - 1)
    scale_factor = ((max_val - min_val) / bits).type(torch.cuda.FloatTensor)

    if scale_factor == 0:
        quant_val = x.type(torch.cuda.CharTensor)
        scale_factor = torch.tensor([1.0]).type(torch.cuda.FloatTensor)
    else:
        #quant_val = torch.round(x / scale_factor).type(torch.cuda.ShortTensor)
        quant_val = torch.clamp(torch.round(x / scale_factor).type(torch.cuda.CharTensor), -bits, bits - 1)
    return quant_val, scale_factor
    
state_dict = torch.load(path)
quant_weight = {}
for key in list(state_dict.keys()):
    if key.startswith('backbone.layers') and key.find('conv') > 0 and key.endswith('weight'):
        print(key, '\tquant\t', state_dict[key].shape)
        q_val, scale = quantizer(state_dict[key])
        error_find_dequant(state_dict[key], q_val, scale)
        quant_weight[key] = [q_val, scale]
    
    elif key.endswith('downsample.0.weight'):
        print(key, '\tquant\t', state_dict[key].shape)
        q_val, scale = quantizer(state_dict[key])
        error_find_dequant(state_dict[key], q_val, scale)
        quant_weight[key] = [q_val, scale]

    elif key.startswith('proto') and key.endswith('weight'):
        print(key, '\tquant\t', state_dict[key].shape)
        q_val, scale = quantizer(state_dict[key])
        error_find_dequant(state_dict[key], q_val, scale)
        quant_weight[key] = [q_val, scale]

    elif key.startswith('fpn') and key.endswith('weight'):
        print(key, '\tquant\t', state_dict[key].shape)
        q_val, scale = quantizer(state_dict[key])
        error_find_dequant(state_dict[key], q_val, scale)
        quant_weight[key] = [q_val, scale]

    elif key.startswith('prediction_layers') and key.endswith('weight'):
        print(key, '\tquant\t', state_dict[key].shape)
        q_val, scale = quantizer(state_dict[key])
        error_find_dequant(state_dict[key], q_val, scale)
        quant_weight[key] = [q_val, scale]

    elif key.endswith('seg_conv.weight'):
        q_val, scale = quantizer(state_dict[key])
        print(key, '\tquant\t', state_dict[key].shape)
        #error_find_dequant(state_dict[key], q_val, scale)
        quant_weight[key] = [q_val, scale]

    elif key.endswith('batches_tracked'):
        print(key, '\tskip\t', state_dict[key].shape)
        pass
    elif key.startswith('backbone.conv1.weight'):
        q_val, scale = quantizer(state_dict[key])
        print(key, '\tquant\t', state_dict[key].shape)
        error_find_dequant(state_dict[key], q_val, scale)
        quant_weight[key] = [q_val, scale]

    elif key.startswith('backbone.conv2.weight'):
        q_val, scale = quantizer(state_dict[key])
        print(key, '\tquant\t', state_dict[key].shape)
        error_find_dequant(state_dict[key], q_val, scale)
        quant_weight[key] = [q_val, scale]
        
    else:
        print(key, '\tno change\t', state_dict[key].shape)
        quant_weight[key] = state_dict[key]

torch.save(quant_weight, os.path.join(args.save, 'yolact_base_0_0.pth')) 


print('Proccess finished')
