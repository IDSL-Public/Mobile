import torch, torchvision
import torch.nn as nn
import argparse 
import numpy as np



parser = argparse.ArgumentParser(description='Prune Yolact')

parser.add_argument('--path', default='weights/', help='Directory for load weight.')
parser.add_argument('--mode', default=0, type=int, help = '1 : mode 1by1 Conv and 3by3 \n0 : Prune all layers')
args = parser.parse_args()

torch.cuda.current_device()


use_jit = torch.cuda.device_count() <= 1
if not use_jit:
    print('Multiple GPUs detected! Turning off JIT.')

ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn

path = args.path
print('path :', path)

    
state_dict = torch.load(path)
# print(state_dict.shape)
if args.mode == 1:
    for key in list(state_dict.keys()):
        print(key,'\t\t',state_dict[key].shape)
elif args.mode == 2:
    for key in list(state_dict.keys()):
        if key.startswith('backbone'):
            if len(state_dict[key]) != 2:
                print(key,'\t\t',state_dict[key].shape, state_dict[key].sum())
        
else:
    for key in list(state_dict.keys()):
        if len(state_dict[key]) == 2:
            print(key,'\t\t',state_dict[key][0].shape, state_dict[key][0].dtype, float(state_dict[key][1]), state_dict[key][1].dtype)
        else:
            print(key,'\t\t',state_dict[key].shape, state_dict[key].sum())

print('Proccess finished')
