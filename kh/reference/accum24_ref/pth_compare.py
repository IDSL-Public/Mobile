import torch, torchvision
import torch.nn as nn
import argparse 
import numpy as np



parser = argparse.ArgumentParser(description='Prune Yolact')

parser.add_argument('--q_path', default='weights/', help='Directory for load weight.')
parser.add_argument('--dq_path', default='weights/', help='Directory for load weight.')
parser.add_argument('--wt_scale_path', default='weights/', help='Directory for load weight.')
parser.add_argument('--act_scale_path', default='weights/', help='Directory for load weight.')
args = parser.parse_args()

torch.cuda.current_device()


use_jit = torch.cuda.device_count() <= 1
if not use_jit:
    print('Multiple GPUs detected! Turning off JIT.')

ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn

def dequantizer(x, scale_factor_a, scale_factor_w):
    dequant_val = x * scale_factor_a * scale_factor_w
    # dequant_val = find_best_approximation(dequant_val, 16)
    return dequant_val.type(torch.FloatTensor)

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

def fixedpoint(x, frac_bit = 16):
    frac_num = x.abs() - x.abs().floor()
    fixed_frac = x * 0
    fixed_int_frac = (x * 0).type(torch.int32)
    int_part = (x.abs() - frac_num).type(torch.int32)
    shifted_int = int_part * 2 ** frac_bit

    for i in range (1, frac_bit+1):
        minus_filter = frac_num - fixed_frac > (2 ** -i)
        comp = (frac_num - (fixed_frac + (2 ** -i))).abs() < (2 ** -frac_bit)
        comp2 = (frac_num - fixed_frac).abs() > (frac_num - (fixed_frac + (2 ** -i))).abs()
        fixed_frac += (minus_filter + torch.logical_and(comp, comp2)) * (2 ** -i)
        fixed_int_frac += (minus_filter + torch.logical_and(comp, comp2)) * (2 ** (frac_bit -i))

    fixed_num = (shifted_int + fixed_int_frac) * x.sign().type(torch.int32)
    fixed_num = np.uint32(fixed_num)
    return fixed_num


quant_val = torch.load(args.q_path).cpu()
dequant_val = torch.load(args.dq_path).cpu()

wt_scale = torch.load(args.wt_scale_path).cpu()
act_scale = torch.load(args.act_scale_path).cpu()

cal_dq_val = dequantizer(quant_val, wt_scale[0], act_scale[0])
fixed_val = find_best_approximation(cal_dq_val)
fixed_data = fixedpoint(fixed_val, 16)
data = fixed_data.reshape(-1).tolist()

print('wt_scaling factor :', wt_scale)
print('act_scaling factor :', act_scale)

print('\nquant val \t', quant_val.shape, '\n', quant_val[0,0,0,:10])
print('\ndequant_val \t', dequant_val.shape, '\n', dequant_val[0,0,:10])
print('\ncal_dq_val \t', cal_dq_val.shape, '\n', cal_dq_val[0,0,0,:10])
print('\nfixed point value \t', fixed_val.shape, '\n', fixed_val[0,0,0,:10])
print('\nfixed point hex value \t', fixed_data.shape, '\n', fixed_data[0,0,0,:10])

print('\nProccess finished')