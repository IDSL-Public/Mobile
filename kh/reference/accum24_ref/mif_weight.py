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


def mif_weight_converter(path, save, name):
    pth_data = torch.load(path).cpu()

    if name == 'name':
        front_num = path.find('pth/')
        front = path[:front_num]
        middle = 'mif/unrolling'
        end = path[front_num+3:-3]

        mif_file_name = front + middle + end + 'mif'
        print('mif_file_name\n', mif_file_name)    
    else:
        mif_file_name = save + name

    if pth_data.type().find('FloatTensor')>0:
        width = 32
        fixed_data = find_best_approximation(pth_data , 16)
        fixed_data = fixedpoint(fixed_data, 16)
        data = fixed_data.reshape(-1).tolist()
        print('FP32 exit()')
        exit()

    elif pth_data.type().find('CharTensor')>0 or pth_data.type().find('ShortTensor')>0:
        width = 8

        if pth_data.shape[3] == 3:

            new_channel = math.ceil(pth_data.shape[1]/16) * 16
            new_data = torch.zeros(pth_data.shape[0], new_channel, pth_data.shape[2], pth_data.shape[3])

            tmp_data = new_data.clone()
            tmp_data[:, :pth_data.shape[1], :, :] = pth_data
            pth_data = tmp_data
            print('before new_data', new_data.shape)
            new_data = new_data.reshape(-1, new_data.shape[0], 3, 16, 3)
            new_data = np.uint8(new_data)
            print('after new_data', new_data.shape)
            uint_out = np.uint8(pth_data)
            
            cnt = 0
            for c in range(pth_data.shape[1]//16):
                for w in range(pth_data.shape[3]):
                    new_data[c,:,w,:,:] = uint_out[:, c*16:(c+1)*16, :, w]
        
        elif pth_data.shape[3] == 1:

            new_channel = math.ceil(pth_data.shape[1]/48) * 48
            new_data = torch.zeros(pth_data.shape[0], new_channel, 1, 1)

            tmp_data = new_data.clone()
            tmp_data[:, :pth_data.shape[1], :, :] = pth_data
            pth_data = tmp_data
            new_data = new_data.reshape(-1, new_data.shape[0], 48, 1, 1)
            new_data = np.uint8(new_data)
            uint_out = np.uint8(pth_data)
            
            cnt = 0
            for c in range(pth_data.shape[1]//48):
                new_data[c,:,:] = uint_out[:, c*48:(c+1)*48, :, :]

        data = new_data.reshape(-1).tolist()

    elif pth_data.type().find('IntTensor')>0:
        width = 32
        uint_out = np.uint32(pth_data)
        data = uint_out.reshape(-1).tolist()
        print('INT32 exit()')
        exit()
    else:
        print('error')
        exit()


    with open(mif_file_name, "w") as mif_file:
        '''
        print('DEPTH = '+ str(len(data)))
        print('WIDTH = '+ str(width))
        print('ADDRESS_RADIX = HEX')
        print('DATA_RADIX = HEX;')
        print('CONTENT BEGIN\n')
        mif_file.write('DEPTH = ' + str(len(data)) + ';\n')
        mif_file.write('WIDTH = ' + str(width) + ';\n')
        mif_file.write('ADDRESS_RADIX = HEX;\n')
        mif_file.write('DATA_RADIX = HEX;\n')
        mif_file.write('CONTENT BEGIN\n')
        '''
        cnt = 0
        for address, value in enumerate(data):
            cnt += 1
            if cnt == 48:
                cnt = 0
                mif_file.write(f"{value:02x}\n")
            else:
                mif_file.write(f"{value:02x}")

mif_weight_converter(args.path, args.save, args.name)

print('Proccess finished')
