import torch
import argparse 
import numpy as np
import os


parser = argparse.ArgumentParser(description='Prune Yolact')

parser.add_argument('--path', default='weights/', help='Directory for load weight.')
args = parser.parse_args()

    
weight = torch.load(args.path)
print('weight :', weight.shape)
