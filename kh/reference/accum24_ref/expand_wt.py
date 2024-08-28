import torch
import argparse 
import numpy as np
import os
import math

parser = argparse.ArgumentParser(description='Prune Yolact')

parser.add_argument('--path', default='weights/', help='Directory for load weight.')
args = parser.parse_args()

    
weight = torch.load(args.path)
print('weight :', weight.shape)
oc = math.ceil(weight.shape[0] / 16) * 16
width = weight.shape[2] // 3
print('width :', width)
print('OC :', oc)
expand_wt = torch.zeros(oc, weight.shape[1], 3, 3).type(torch.cuda.CharTensor)
print('weight.sum()', weight.sum())
if width == 1:
    expand_wt[:weight.shape[0]] = weight
    torch.save(expand_wt, os.path.join('./origin_data/sangbom_rq/pth/L92_CONV_WT_INT8_EXP.pth'))
else:
    expand_wt[:weight.shape[0], :, 1, 1] = weight.squeeze(3).squeeze(2)
    print('expand_wt :', expand_wt.shape)
    print('expand_wt[0,1,:,:]', expand_wt[0,1,:,:])
    torch.save(expand_wt, os.path.join('./origin_data/sangbom_rq/pth/L92_CONV_WT_INT8_EXP.pth'))    
    print('expanded weight sum :', expand_wt.sum())

print('expanded weight :', expand_wt.shape)
weight = expand_wt.permute(1,0,2,3)
print('premuted weight :', weight.shape)
torch.save(weight, os.path.join('./origin_data/sangbom_rq/pth/L92_CONV_WT_INT8_PER.pth'))

rotate_weight = weight.rot90(2, [2, 3])
torch.save(rotate_weight, os.path.join('./origin_data/sangbom_rq/pth/L92_CONV_WT_INT8_PER_ROT.pth'))