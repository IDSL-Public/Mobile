import os
import subprocess
import argparse
import torch, torchvision
import torch.nn as nn
import argparse
import numpy as np
import math

parser = argparse.ArgumentParser(description='pth to mif converter')
parser.add_argument('--type', default='weight', help='Directory for load weight.')
parser.add_argument('--path', default='weight', help='Directory for load weight.')
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

def mif_weight_converter(path):
    pth_data = torch.load(path).cpu()

    front_num = path.find('pth/')
    front = path[:front_num]
    middle = 'mif'
    end = path[front_num+3:-3]
    print('\npth_data type', pth_data.type())

    mif_file_name = front + middle + end + 'mif'
    print('mif_file_name\n', mif_file_name)    

    

    if pth_data.type().find('FloatTensor')>0:
        fixed_data = find_best_approximation(pth_data , 16)
        fixed_data = fixedpoint(fixed_data, 16)
        data = fixed_data.reshape(-1).tolist()

        with open(mif_file_name, "w") as mif_file:
            for address, value in enumerate(data):
                mif_file.write(f"{value:08x}\n")

    elif pth_data.type().find('CharTensor')>0 or pth_data.type().find('ShortTensor')>0:
        width = 8

        if pth_data.shape[3] == 3:

            new_channel = math.ceil(pth_data.shape[1]/16) * 16
            new_filter = math.ceil(pth_data.shape[0]/16) * 16
            new_data = torch.zeros(new_filter, new_channel, pth_data.shape[2], pth_data.shape[3])

            new_data[:pth_data.shape[0], :pth_data.shape[1]] = pth_data
            oc_unroll = math.ceil(new_data.shape[0] / 16)
            ic_unroll = math.ceil(new_data.shape[1] / 16)
            print('new_data.shape :', new_data.shape)
            print('oc_unroll :', oc_unroll)
            print('ic_unroll :', ic_unroll)

            print('new_data :', new_data.shape)
            new_data = new_data.reshape(oc_unroll, 16, ic_unroll, 16, 3, 3)
            new_data = new_data.permute(0, 2, 1, 5, 3, 4).reshape(-1)
            new_data = np.uint8(new_data)

            # tmp_data = new_data.clone()
            # tmp_data[:, :pth_data.shape[1], :, :] = pth_data
            # pth_data = tmp_data
            # # print('before new_data', new_data.shape)
            # new_data = new_data.reshape(-1, new_data.shape[0], 3, 16, 3)
            # new_data = np.uint8(new_data)
            # # print('after new_data', new_data.shape)
            # uint_out = np.uint8(pth_data)
            
            # cnt = 0
            # for c in range(pth_data.shape[1]//16):
            #     for w in range(pth_data.shape[3]):
            #         new_data[c,:,w,:,:] = uint_out[:, c*16:(c+1)*16, :, w] 
        
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
            for _, value in enumerate(data):
                cnt += 1
                if cnt == 48:
                    cnt = 0
                    mif_file.write(f"{value:02x}\n")
                else:
                    mif_file.write(f"{value:02x}")


    elif pth_data.type().find('IntTensor')>0:
        width = 32
        uint_out = np.uint32(pth_data)
        data = uint_out.reshape(-1).tolist()
        print('INT32 pass')
        pass
    else:
        print('error')
        pass


def mif_scale_converter(path):
    pth_data = torch.load(path).cpu()
    print('\npth_data type', pth_data.type())

    front_num = path.find('pth/')
    front = path[:front_num]
    middle = 'mif'
    end = path[front_num+3:-3]

    mif_file_name = front + middle + end + 'mif'
    print('mif_file_name\n', mif_file_name)    

    if pth_data.type().find('FloatTensor')>0:
        fixed_data = find_best_approximation(pth_data , 16)
        fixed_data = fixedpoint(fixed_data, 16)
        data = fixed_data.reshape(-1).tolist()
        
        with open(mif_file_name, "w") as mif_file:
            for _, value in enumerate(data):
                mif_file.write(f"{value:08x}\n")

    else:
        print(pth_data.type(), '\terror')
        pass

def mif_act_converter(path):
    pth_data = torch.load(path).cpu()
    print('\npth_data type', pth_data.type())

    front_num = path.find('pth/')
    front = path[:front_num]
    middle = 'mif'
    end = path[front_num+3:-3]

    mif_file_name = front + middle + end + 'mif'
    print('mif_file_name\n', mif_file_name)    

    if pth_data.type().find('FloatTensor')>0:

        fixed_data = find_best_approximation(pth_data , 16)

        new_channel = math.ceil(fixed_data.shape[1]/32) * 32
        new_data = torch.zeros(fixed_data.shape[0], new_channel, fixed_data.shape[2], fixed_data.shape[3])

        tmp_data = new_data.clone()
        tmp_data[:, :fixed_data.shape[1], :, :] = fixed_data
        fixed_data = tmp_data
        new_data = new_data.reshape(new_data.shape[0], -1, 32)
        new_data = np.uint32(new_data)
        fixed_data = fixedpoint(fixed_data, 16)
        uint_out = np.uint32(fixed_data)
        cnt = 0
        for b in range(fixed_data.shape[0]):
            for c in range(fixed_data.shape[1]//32):
                for h in range(fixed_data.shape[2]):
                    for w in range(fixed_data.shape[3]):
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

    elif pth_data.type().find('CharTensor')>0 or pth_data.type().find('ShortTensor')>0:

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


    elif pth_data.type().find('IntTensor')>0 or pth_data.type().find('LongTensor')>0:
        width = 32

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
        pass


def convert_pth_to_mif(pth_path, type):
    # 여기에 .pth 파일을 .mif 파일로 변환하는 코드를 구현하세요.
    # 예: subprocess.run(['python', 'convert_script.py', '--path', pth_path])
    if type == 'weight':
        mif_weight_converter(pth_path)
    elif type == 'act':
        mif_act_converter(pth_path)
    elif type == 'scale':
        mif_scale_converter(pth_path)

def find_and_convert_files(directory, type):
    # 지정된 디렉토리 내의 모든 .pth 파일을 찾습니다.
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pth"):
                pth_path = os.path.join(root, file)
                # 각 .pth 파일에 대해 변환 함수를 호출합니다.
                convert_pth_to_mif(pth_path, type)
                print(f"Converted: {pth_path}")

# 변환을 시작할 디렉토리 경로를 지정하세요.
start_directory = args.path

if start_directory.find('act') > 0:
    auto_type = 'act'
elif start_directory.find('scale') > 0:
    auto_type = 'scale'
elif start_directory.find('weight') > 0:
    auto_type = 'weight'
else:
    auto_type = args.type

find_and_convert_files(start_directory, auto_type)