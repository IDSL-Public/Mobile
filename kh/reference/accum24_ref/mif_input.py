import torch, torchvision
import torch.nn as nn
import argparse
import numpy as np
import math


parser = argparse.ArgumentParser(description='Prune Yolact')

parser.add_argument('--path', default='weights/', help='Directory for load weight.')
parser.add_argument('--save', default='weights/', help='Directory for load weight.')
parser.add_argument('--name', default='mif.mif', help='Directory for load weight.')

args = parser.parse_args()

path = args.path
    
pth_data = torch.load(path).cpu()
print('pth_data', pth_data.shape)

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


def mif_act_converter(path, save, name):
    pth_data = torch.load(path).cpu()

    if name == 'name':
        front_num = path.find('pth/')
        front = path[:front_num]
        middle = 'mif'
        end = path[front_num+3:-3]

        mif_file_name = front + middle + end + 'mif'
        print('\nmif_file_name\n', mif_file_name)    
    else:
        mif_file_name = save + name

    if pth_data.type().find('FloatTensor')>0:
        width = 32
        fixed_data = find_best_approximation(pth_data , 16)
        fixed_data = fixedpoint(fixed_data, 16)
        data = fixed_data.reshape(-1).tolist()
        exit()

    elif pth_data.type().find('CharTensor')>0 or pth_data.type().find('ShortTensor')>0:
        width = 8

        new_channel = math.ceil(pth_data.shape[1]/16) * 16
        new_height = math.ceil(pth_data.shape[2]/3) * 3
        new_data = torch.zeros(pth_data.shape[0], new_channel, new_height, pth_data.shape[3])

        tmp_data = new_data.clone()
        tmp_data[:, :pth_data.shape[1], :pth_data.shape[2], :] = pth_data
        pth_data = tmp_data
        new_data = new_data.reshape(new_data.shape[0], -1, 16, 3)
        new_data = np.uint8(new_data)
        uint_out = np.uint8(pth_data)
        cnt = 0
        for b in range(pth_data.shape[0]):
            for c in range(pth_data.shape[1]//16):
                for h in range(pth_data.shape[2]//3):
                    for w in range(pth_data.shape[3]):
                        new_data[b, cnt] = uint_out[b, c*16:(c+1)*16, h*3:(h+1)*3, w]
                        if cnt == new_data.shape[1]-1:
                            cnt = 0
                        else:
                            cnt += 1
        
        data = new_data.reshape(-1).tolist()

        with open(mif_file_name, "w") as mif_file:
            w_cnt = 0
            for _, value in enumerate(data):
                w_cnt += 1
                if w_cnt == 48:
                    w_cnt = 0
                    mif_file.write(f"{value:02x}\n")
                else:
                    mif_file.write(f"{value:02x}")


    elif pth_data.type().find('IntTensor')>0:
        width = 32

        
        # pth_data = torch.zeros(2, 128, 5, 5).type(torch.IntTensor)
        # print('pth_data', pth_data.shape)
        # d_cnt = 0
        # for i in range(128//32):
        #     for h in range(pth_data.shape[2]):
        #         for w in range(pth_data.shape[3]):
        #             pth_data[:, i*32:(i+1)*32, h, w] = d_cnt
        #             d_cnt += 1
       

        new_channel = math.ceil(pth_data.shape[1]/32) * 32
        new_data = torch.zeros(pth_data.shape[0], new_channel, pth_data.shape[2], pth_data.shape[3])

        tmp_data = new_data.clone()
        tmp_data[:, :pth_data.shape[1], :, :] = pth_data
        pth_data = tmp_data
        new_data = new_data.reshape(new_data.shape[0], -1, 32)
        new_data = np.uint32(new_data)
        uint_out = np.uint32(pth_data)
        cnt = 0
        for b in range(pth_data.shape[0]):
            for c in range(pth_data.shape[1]//32):
                for h in range(pth_data.shape[2]):
                    for w in range(pth_data.shape[3]):
                        new_data[b, cnt] = uint_out[b, c*32:(c+1)*32, h, w]
                        if cnt == new_data.shape[1]-1:
                            cnt = 0
                        else:
                            cnt += 1

        data = new_data.reshape(-1).tolist()
        
        with open(mif_file_name, "w") as mif_file:
            cnt = 0
            for _, value in enumerate(data):
                cnt += 1
                if cnt == 32:
                    cnt = 0
                    mif_file.write(f"{value:08x}\n")
                else:
                    mif_file.write(f"{value:08x}")

    else:
        print('error')
        exit()

    

mif_act_converter(args.path, args.save, args.name)

print('Proccess finished')

