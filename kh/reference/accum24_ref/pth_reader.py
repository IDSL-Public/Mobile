import torch, torchvision
import torch.nn as nn
import argparse 
import numpy as np



parser = argparse.ArgumentParser(description='Prune Yolact')

parser.add_argument('--path1', default='weights/', help='Directory for load weight.')
parser.add_argument('--path2', default='weights/', help='Directory for load weight.')
args = parser.parse_args()

    
val1 = torch.load(args.path1)
val2 = torch.load(args.path2)

print(args.path1)
print(args.path2)

print(val1)
print(val2)
