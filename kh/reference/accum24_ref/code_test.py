import numpy as np
import torch
import copy
from components import *
import torch.nn.functional as F
 
a = torch.randn(3, 3, 224, 224).cuda()
conv = conv_layer(3, 64, 3, 3, stride=1, shift=False)

print('--------------------conv_layer----------------------')
print('conv input type = ', a.type())
print('conv weight type = ', conv.weight.type())
print('conv output type = ', conv.forward(a).type())


print('--------------------linear_layer---------------------')

q = torch.randn(128, 20).cuda()
linear = fc_sigmoid(20, 30)

print('linear input type = ', q.type())
print('linear weight type = ', linear.kernel.type())
print('linear output type = ', linear.forward(q).type())

print('--------------------maxpool_layer---------------------')

w = torch.randn(3, 3, 224, 224).cuda()
pool = max_pooling(3, 3, stride=2)

print('maxpool input type = ', w.type())
print('maxpool output type = ', pool.forward(w).type())

print('--------------------avgpool_layer---------------------')
e = torch.randn(3, 3, 224, 224).cuda()
avgpool = global_average_pooling()

print('avgpool input type = ', e.type())
print('avgpool output type = ', avgpool.forward(e).type())

print('----------------------relu_layer-----------------------')

r = torch.randn(3, 3, 224, 224).cuda()
relu = relu()

print('relu input type = ', r.type())
print('relu output type = ', relu.forward(r).type())

print('----------------------bn_layer-----------------------')

t = torch.randn(3, 64, 32, 32).cuda()
bn = bn_layer(64)

print('bn input type = ', t.type())
print('bn output type = ', bn.forward(t).type())

print('----------------------tanh_layer-----------------------')
z = torch.randn(3, 64, 32, 32).cuda()
tanh = tanh()

print('tanh input type = ', z.type())
print('tanh output type = ', tanh.forward(z).type())



