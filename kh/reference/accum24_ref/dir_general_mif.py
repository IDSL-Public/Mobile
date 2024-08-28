import os
import subprocess
import argparse
import torch, torchvision
import torch.nn as nn
import argparse
import numpy as np
import math

parser = argparse.ArgumentParser(description='pth to mif converter')
parser.add_argument('--path', default='weight', help='Directory for load weight.')
args = parser.parse_args()


args = parser.parse_args()

def find_best_approximation(value, max_n=10):
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

def mif_converter(path):
    pth_data = torch.load(path).cpu()
    front_num = path.find('pth/')
    front = path[:front_num]
    middle = 'mif'
    end = path[front_num+3:-3]

    mif_file_name = front + middle + end + 'mif'
    print('\nmif_file_name\n', mif_file_name)    
    
    if pth_data.type().find('FloatTensor')>0:
        width = 32
        fixed_data = find_best_approximation(pth_data , 16)
        fixed_data = fixedpoint(fixed_data, 16)
        data = fixed_data.reshape(-1).tolist()
    
    elif pth_data.type().find('CharTensor')>0 or pth_data.type().find('ShortTensor')>0:
        width = 8
        uint_out = np.uint8(pth_data)
        data = uint_out.reshape(-1).tolist()
    elif pth_data.type().find('IntTensor')>0 or pth_data.type().find('LongTensor')>0:
        width = 32
        uint_out = np.uint32(pth_data)
        data = uint_out.reshape(-1).tolist()
    else:
        print('error')
        exit()
    
    with open(mif_file_name, "w") as mif_file:
        if width == 8:
            for address, value in enumerate(data):
                mif_file.write(f"{value:02x}\n")
        else:
            for address, value in enumerate(data):
                mif_file.write(f"{value:08x}\n")


def find_and_convert_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pth"):
                pth_path = os.path.join(root, file)
                mif_converter(pth_path)
                print(f"Converted: {pth_path}")

start_directory = args.path
find_and_convert_files(start_directory)