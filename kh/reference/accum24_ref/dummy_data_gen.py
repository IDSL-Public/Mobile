import os
import subprocess
import argparse
import torch, torchvision
import torch.nn as nn
import argparse
import numpy as np
import math

data = torch.ones(20,64,1,1).type(torch.cuda.CharTensor)
# data = torch.ones(64,32,3,3).type(torch.cuda.IntTensor)

r = 0
# for i in range(64):
#     # if i % 5 == 0:
#     #     r = 0
for j in range(64):
    data[:, j, :, :] = j
    # r += 1
# print(data[0,:,:,:])

torch.save(data, os.path.join('./origin_data/sangbom_rq/pth/common_layer_20x64x1x1.pth'))

# data = torch.ones(20, 12, 3, 3).type(torch.cuda.CharTensor)

# for j in range(12):
#     for k in range(3):
#         data[:, j, :, k] = j * 3 + k

# torch.save(data, os.path.join('./origin_data/dummy/pth/uncommon_layer_20x12x3x3.pth'))


# data = torch.ones(16, 32, 1, 1).type(torch.cuda.CharTensor)

# for j in range(32):
#     data[:, j] = j

# for f in range(16):
#     data[f] += f

# torch.save(data, os.path.join('./origin_data/dummy/pth/common_layer_16x32x1x1.pth'))


# data = torch.ones(64, 3, 3, 3).type(torch.cuda.CharTensor)

# for k in range(3):
#     for h in range(3):
#         for w in range(3):
#             data[:, k, h, w] = k * 9 + h * 3 + w

# for f in range(64):
#     data[f] += f

# torch.save(data, os.path.join('./origin_data/dummy/pth/first_layer_64x3x3x3.pth'))




# f_unit = data.shape[0] // 16
# k_unit = data.shape[1] // 16

# print(data[0,0])
# print(data[1,0])

# data = data.reshape(f_unit, 16, k_unit, 16, 3, 3)
# print('data.shape :', data.shape)
# data = data.permute(0, 1, 2, 5, 3, 4)
# data = data.reshape(data.shape[0], data.shape[1], data.shape[2], data.shape[3], -1)
# # print(data[0,0,0])
# print('data.shape :', data.shape)
# print(data[0,0,0])
# print(data[0,0,1])
# data = data.permute(0,2,1,3,4)
# print()
# print(data[0,0,0])
# print(data[0,0,1])
# print('data.shape :', data.shape)
