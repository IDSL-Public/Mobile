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

def dequantizer(x, scale_factor_w):
    dequant_val = x * scale_factor_w
    return dequant_val.type(torch.cuda.FloatTensor)
    
state_dict = torch.load(path)
quant_weight = {}
for key in list(state_dict.keys()):
    if key.startswith('backbone.layers') and key.find('conv') > 0 and key.endswith('weight'):
        print(key, '\tdequant\t', state_dict[key][0].shape)
        q_val = dequantizer(state_dict[key][0], state_dict[key][1])
        quant_weight[key] = q_val
    elif key.endswith('downsample.0.weight'):
        print(key, '\tdequant\t', state_dict[key][0].shape)
        q_val = dequantizer(state_dict[key][0], state_dict[key][1])
        quant_weight[key] = q_val
    elif key.startswith('proto') and key.endswith('weight'):
        print(key, '\tdequant\t', state_dict[key][0].shape)
        q_val = dequantizer(state_dict[key][0], state_dict[key][1])
        quant_weight[key] = q_val
    elif key.startswith('fpn') and key.endswith('weight'):
        print(key, '\tdequant\t', state_dict[key][0].shape)
        q_val = dequantizer(state_dict[key][0], state_dict[key][1])
        quant_weight[key] = q_val
    elif key.startswith('prediction_layers') and key.endswith('weight'):
        print(key, '\tdequant\t', state_dict[key][0].shape)
        q_val = dequantizer(state_dict[key][0], state_dict[key][1])
        quant_weight[key] = q_val
    elif key.endswith('seg_conv.weight'):
        print(key, '\tdequant\t', state_dict[key][0].shape)
        q_val = dequantizer(state_dict[key][0], state_dict[key][1])
        quant_weight[key] = q_val
    elif key.endswith('batches_tracked'):
        print(key, '\tskip\t', state_dict[key].shape)
        pass
    elif key.startswith('backbone.conv1.weight'):
        print(key, '\tdequant\t', state_dict[key][0].shape)
        q_val = dequantizer(state_dict[key][0], state_dict[key][1])
        quant_weight[key] = q_val
    elif key.startswith('backbone.conv2.weight'):
        print(key, '\tdequant\t', state_dict[key][0].shape)
        q_val = dequantizer(state_dict[key][0], state_dict[key][1])
        quant_weight[key] = q_val
    else:
        print(key, '\tno change\t', state_dict[key].shape)
        quant_weight[key] = state_dict[key]

torch.save(quant_weight, os.path.join(args.save, 'yolact_base_0_0.pth')) 


print('Proccess finished')
