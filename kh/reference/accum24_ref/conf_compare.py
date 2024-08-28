import torch, torchvision
import torch.nn as nn
import argparse 
import numpy as np



parser = argparse.ArgumentParser(description='Prune Yolact')

parser.add_argument('--pth_path', default='weights/', help='Directory for load weight.')
# parser.add_argument('--q_path', default='weights/', help='Directory for load weight.')
parser.add_argument('--dq_path', default='weights/', help='Directory for load weight.')
# parser.add_argument('--wt_scale_path', default='weights/', help='Directory for load weight.')
# parser.add_argument('--act_scale_path', default='weights/', help='Directory for load weight.')
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

def hex_to_decimal(hex_str):
    num = []
    for i in range(len(hex_str)):
        num.append(int(hex_str[i], 16))

    int_val = np.int32(num)
    sign_bit = np.sign(int_val)

    int_val *= sign_bit

    integer_part = (int_val >> 16) & 0x7FFF  # 15 bits for integer part (excluding sign bit)
    fractional_part = int_val & 0xFFFF       # 16 bits for fractional part

    integer_value = integer_part

    fractional_value = 0
    for i in range(16):
        fractional_value += ((fractional_part >> (15 - i)) & 1) * (2 ** -(i + 1))
    decimal_value = sign_bit * (integer_value + fractional_value)

    return decimal_value

def hex_to_decimal_2s_complement(hex_str):
    num = []
    for i in range(len(hex_str)):
        num.append(int(hex_str, 16))
    sign_bit = (num >> 31) & 1
    if sign_bit == 1:
        num = -((~num + 1) & 0xFFFFFFFF)
    return num


conv_out_fp = np.array(torch.load(args.pth_path).cpu().reshape(-1))

with open(args.dq_path, 'r', encoding='utf-8') as file:
    conv_dq_out_mif = file.readlines()
conv_dq_out_mif = [line.strip() for line in conv_dq_out_mif]

# with open(args.q_path, 'r', encoding='utf-8') as file:
#     conv_int_out = file.readlines()
# conv_int_out = [line.strip() for line in conv_int_out]

# with open(args.wt_scale_path, 'r', encoding='utf-8') as file:
#     wt_scale = file.readlines()
# wt_scale = [line.strip() for line in wt_scale]

# with open(args.act_scale_path, 'r', encoding='utf-8') as file:
#     act_scale = file.readlines()
# act_scale = [line.strip() for line in act_scale]

# print('wt_scale :', wt_scale)
# print('act_scale :', act_scale)

# wt_scale = hex_to_decimal(wt_scale)
# act_scale = hex_to_decimal(act_scale)

conv_dq_out_mif = hex_to_decimal(conv_dq_out_mif)
# conv_int_out = hex_to_decimal_2s_complement(conv_int_out)
# dq_conv_int_out = conv_int_out * wt_scale * act_scale

# print('dequantized value \n', dq_conv_int_out[:10])
# print('conv_dq_out_mif \n', conv_dq_out_mif[:10])
# print('conv_out_fp \n', conv_out_fp[:10])

# print(((conv_dq_out_mif[:10] - conv_out_fp[:10]) / conv_out_fp[:10]) * 100)

difference = conv_out_fp - conv_dq_out_mif
rate = (difference / conv_out_fp) * 100

shape_check = args.dq_path.find('x')
print('shape check',args.dq_path[shape_check+1])

if args.dq_path[shape_check + 1] == '6':
    diff_path = './output_folder/CONF_CONV_OUT_DQ_DIFF_69x69.txt'
    rate_path = './output_folder/CONF_CONV_OUT_DQ_RATIO_69x69.txt'
elif args.dq_path[shape_check + 1] == '3':
    diff_path = './output_folder/CONF_CONV_OUT_DQ_DIFF_35x35.txt'
    rate_path = './output_folder/CONF_CONV_OUT_DQ_RATIO_35x35.txt'
elif args.dq_path[shape_check + 1] == '1':
    diff_path = './output_folder/CONF_CONV_OUT_DQ_DIFF_18x18.txt'
    rate_path = './output_folder/CONF_CONV_OUT_DQ_RATIO_18x18.txt'
elif args.dq_path[shape_check + 1] == '9':
    diff_path = './output_folder/CONF_CONV_OUT_DQ_DIFF_9x9.txt'
    rate_path = './output_folder/CONF_CONV_OUT_DQ_RATIO_9x9.txt'
elif args.dq_path[shape_check + 1] == '5':
    diff_path = './output_folder/CONF_CONV_OUT_DQ_DIFF_5x5.txt'
    rate_path = './output_folder/CONF_CONV_OUT_DQ_RATIO_5x5.txt'

with open(diff_path, 'w', encoding='utf-8') as file:
    for element in difference:
        file.write(f"{element}\n")

with open(rate_path, 'w', encoding='utf-8') as file:
    for element in rate:
        file.write(f"{element}%\n")