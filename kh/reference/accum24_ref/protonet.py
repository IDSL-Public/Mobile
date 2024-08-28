import numpy as np
import torch
import copy
from components import *
import torch.nn.functional as F
import torch
#from customconv2d import customConv2d
from customReLU import customReLU
import time
import tracemalloc

class Protonet_class:
    def __init__(self):
        self.layer = [
            conv_layer(256, 128, 3, 3, stride = 1, shift=True, quant=True),
            relu(),
            conv_layer(128, 128, 3, 3, stride = 1, shift=True, quant=True),
            relu(),
            conv_layer(128, 128, 3, 3, stride = 1, shift=True, quant=True),
            relu(),
            relu(),
            conv_layer(128, 128, 3, 3, stride = 1, shift=True, quant=True),
            relu(),
            conv_layer(128, 32, 1, 1, stride = 1, shift=True, pad=0, quant=True)
            ]
        self.upsample = upsample('proto')
        self.cnt = 0
    
    def forward(self, in_tensor):
        #print('ProtoNet FW Start!')
        # x1 = in_tensor
        # torch.save(x1, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/PROTONET_INPUT.pth'))

        # x1 = self.layer[0].forward(x1) #CONV0 (128, 256, 3, 3) BIAS = True Stride = 1
        # # torch.save(self.layer[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L82_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer[0].act_scale_mean[-1], 1/self.layer[0].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L82_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(torch.tensor([self.layer[0].in_scale[-1], 1/self.layer[0].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/scale/L82_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer[0].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L82_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer[0].conv_dq_out[-1], os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L82_CONV_OUT_DQ_FP32.pth'))
        # # torch.save(x1, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L82_CONV_OUT_BIASED_DQ_FP32.pth'))
        # # torch.save(self.layer[0].weight, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/weight/L82_CONV_WT_INT8.pth'))
        # # torch.save(self.layer[0].bias, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/weight/L82_CONV_BIAS_FP32.pth'))
        # # torch.save(torch.tensor([self.layer[0].wt_scale, 1/self.layer[0].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/scale/L82_CONV_WT_SCALE_FP32.pth'))
        # x1 = self.layer[1].forward(x1) #RELU1
        # # torch.save(x1, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L83_RELU_OUT.pth'))
        
        # x1 = self.layer[2].forward(x1) #CONV2 (128, 128, 3, 3) BIAS = True Stride = 1
        # # torch.save(self.layer[2].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L84_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer[2].act_scale_mean[-1], 1/self.layer[2].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L84_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(torch.tensor([self.layer[2].in_scale[-1], 1/self.layer[2].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/scale/L84_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer[2].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L84_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer[2].conv_dq_out[-1], os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L84_CONV_OUT_DQ_FP32.pth'))
        # # torch.save(x1, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L84_CONV_OUT_BIASED_DQ_FP32.pth'))
        # # torch.save(self.layer[2].weight, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/weight/L84_CONV_WT_INT8.pth'))
        # # torch.save(self.layer[2].bias, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/weight/L84_CONV_BIAS_FP32.pth'))
        # # torch.save(torch.tensor([self.layer[2].wt_scale, 1/self.layer[2].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/scale/L84_CONV_WT_SCALE_FP32.pth'))
        # x1 = self.layer[3].forward(x1) #RELU3
        # # torch.save(x1, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L85_RELU_OUT.pth'))
        
        # x1 = self.layer[4].forward(x1) #CONV4 (128, 128, 3, 3) BIAS = True Stride = 1
        # # torch.save(self.layer[4].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L86_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer[4].act_scale_mean[-1], 1/self.layer[4].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L86_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(torch.tensor([self.layer[4].in_scale[-1], 1/self.layer[4].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/scale/L86_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer[4].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L86_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer[4].conv_dq_out[-1], os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L86_CONV_OUT_DQ_FP32.pth'))
        # # torch.save(x1, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L86_CONV_OUT_BIASED_DQ_FP32.pth'))
        # # torch.save(self.layer[4].weight, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/weight/L86_CONV_WT_INT8.pth'))
        # # torch.save(self.layer[4].bias, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/weight/L86_CONV_BIAS_FP32.pth'))
        # # torch.save(torch.tensor([self.layer[4].wt_scale, 1/self.layer[4].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/scale/L86_CONV_WT_SCALE_FP32.pth'))
        # x1 = self.layer[5].forward(x1) #RELU5
        # # torch.save(x1, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L87_RELU_OUT_FP32.pth'))

        # x1 = self.upsample.forward(x1)
        # # torch.save(x1, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L88_UPSAMPLE_OUT_FP32.pth'))

        # x1 = self.layer[6].forward(x1) #RELU7
        # # torch.save(x1, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L89_RELU_OUT_FP32.pth'))
        
        # x1 = self.layer[7].forward(x1) #CONV8 (128, 128, 3, 3) BIAS = True Stride = 1
        # # torch.save(self.layer[7].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L90_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer[7].act_scale_mean[-1], 1/self.layer[7].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L90_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(torch.tensor([self.layer[7].in_scale[-1], 1/self.layer[7].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/scale/L90_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer[7].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L90_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer[7].conv_dq_out[-1], os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L90_CONV_OUT_DQ_FP32.pth'))
        # # torch.save(x1, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L90_CONV_OUT_BIASED_DQ_FP32.pth'))
        # # torch.save(self.layer[7].weight, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/weight/L90_CONV_WT_INT8.pth'))
        # # torch.save(self.layer[7].bias, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/weight/L90_CONV_BIAS_FP32.pth'))
        # # torch.save(torch.tensor([self.layer[7].wt_scale, 1/self.layer[7].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/scale/L90_CONV_WT_SCALE_FP32.pth'))
        # x1 = self.layer[8].forward(x1) #RELU9
        # # torch.save(x1, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L91_RELU_OUT.pth'))

        # x1 = self.layer[9].forward(x1) #CONV10 (128, 32, 1, 1) BIAS = True Stride = 1
        # # torch.save(self.layer[9].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L92_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer[9].act_scale_mean[-1], 1/self.layer[9].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L92_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(torch.tensor([self.layer[9].in_scale[-1], 1/self.layer[9].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/scale/L92_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer[9].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L92_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer[9].conv_dq_out[-1], os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L92_CONV_OUT_DQ_FP32.pth'))
        # # torch.save(self.layer[9].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/act/L92_CONV_OUT_BIASED_DQ_FP32.pth'))
        # # torch.save(self.layer[9].weight, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/weight/L92_CONV_WT_INT8.pth'))
        # # torch.save(self.layer[9].bias, os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/weight/L92_CONV_BIAS_FP32.pth'))
        # # torch.save(torch.tensor([self.layer[9].wt_scale, 1/self.layer[9].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/PROTONET/scale/L92_CONV_WT_SCALE_FP32.pth'))

        
        x1 = in_tensor
        x1 = self.layer[0].forward(x1) #CONV0 (128, 256, 3, 3) BIAS = True Stride = 1
        x1 = self.layer[1].forward(x1) #RELU1
        x1 = self.layer[2].forward(x1) #CONV2 (128, 128, 3, 3) BIAS = True Stride = 1
        x1 = self.layer[3].forward(x1) #RELU3
        x1 = self.layer[4].forward(x1) #CONV4 (128, 128, 3, 3) BIAS = True Stride = 1
        x1 = self.layer[5].forward(x1) #RELU5
        x1 = self.upsample.forward(x1)
        x1 = self.layer[6].forward(x1) #RELU7
        x1 = self.layer[7].forward(x1) #CONV8 (128, 128, 3, 3) BIAS = True Stride = 1
        x1 = self.layer[8].forward(x1) #RELU9
        x1 = self.layer[9].forward(x1) #CONV10 (128, 32, 1, 1) BIAS = True Stride = 1
        
        return x1
        
    def load(self, path):
        weight = torch.load(path)
        
        proto_weight = []
        proto_bias = []
        
        for key in list(weight.keys()):
            if key.startswith('proto_net') and key.endswith('weight'):
                proto_weight.append(weight[key])
            elif key.startswith('proto_net') and key.endswith('bias'):
                proto_bias.append(weight[key])
        
        self.layer[0].load([proto_weight[0], proto_bias[0]])
        self.layer[2].load([proto_weight[1], proto_bias[1]])
        self.layer[4].load([proto_weight[2], proto_bias[2]])
        self.layer[7].load([proto_weight[3], proto_bias[3]])
        self.layer[9].load([proto_weight[4], proto_bias[4]])
        
        #for i in range(5):
        #    self.layer[i*2].load([proto_weight[i], proto_bias[i]])
            
    def backward(self, lr, weight_decay, momentum):
        
        #print('ProtoNet BW Process start!')
        #Torch Proto Loss
        #loss = torch.load('/home/kh/reference/pascal_ref_yolact/loss/proto_relu_loss.pth')
        #Ref Proto Loss
        loss = torch.load('./ref_loss/proto_loss.pth')

        loss = self.layer[9].backward(loss, lr, weight_decay, momentum) #Conv8 Layer Backward
        #print('ProtoNet Conv10') 
        #self.layer[9].monitor()
        loss = self.layer[8].backward(loss)                             #ReLU Layer Backward
        #print('ProtoNet ReLU9')
        #self.layer[8].monitor()
        loss = self.layer[7].backward(loss, lr, weight_decay, momentum) #Conv6 Layer Backward
        #print('ProtoNet Conv8')
        #self.layer[7].monitor()

        loss = self.layer[6].backward(loss)                             #ReLU Layer Backward
        #print('ProtoNet ReLU 7')
        #self.layer[6].monitor()
        loss = self.upsample.backward(loss)                          #Interpolate Layer Backward
        #self.upsample.monitor()
        loss = self.layer[5].backward(loss)                             #ReLU Layer Backward
        #print('ProtoNet ReLU 5')
        #self.layer[5].monitor()

        loss = self.layer[4].backward(loss, lr, weight_decay, momentum) #Conv4 Layer Backward
        #print('ProtoNet Conv4')
        #self.layer[4].monitor()
        loss = self.layer[3].backward(loss)                             #ReLU Layer Backward
        loss = self.layer[2].backward(loss, lr, weight_decay, momentum) #Conv2 Layer Backward
        #print('ProtoNet Conv2')
        #self.layer[2].monitor()
        loss = self.layer[1].backward(loss)                             #ReLU Layer Backward
        loss = self.layer[0].backward(loss, lr, weight_decay, momentum) #Conv0 Layer Backward
        #print('ProtoNet Conv0')
        #self.layer[0].monitor()

        
        return loss







'''
import numpy as np
import torch
import copy
from components import *
import torch.nn.functional as F
import torch
#from customconv2d import customConv2d
from customReLU import customReLU

class Protonet_class:
    def __init__(self):
        self.layer = [
            conv_layer(256, 256, 3, 3, stride = 1, shift=True),
            relu(),
            conv_layer(256, 256, 3, 3, stride = 1, shift=True),
            relu(),
            conv_layer(256, 256, 3, 3, stride = 1, shift=True),
            relu(),
            conv_layer(256, 256, 3, 3, stride = 1, shift=True),
            relu(),
            conv_layer(256, 32, 1, 1, stride = 1, shift=True, pad=0)
            ]
        self.interpolate = interpolate
    
    def forward(self, in_tensor):
        x1 = copy.deepcopy(in_tensor)
        cnt=1
        x1 = self.layer[0].forward(x1)
        x1 = self.layer[1].forward(x1)
        
        x1 = self.layer[2].forward(x1)
        x1 = self.layer[3].forward(x1)
        
        x1 = self.layer[4].forward(x1)
        x1 = self.layer[5].forward(x1)
        
        #print('ProtoNet interpolate')
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False)
        #x1 = self.interpolate.forward(x1)
        
        x1 = self.layer[6].forward(x1)
        x1 = self.layer[7].forward(x1)
        x1 = self.layer[8].forward(x1)

        return x1
        
    def load(self, path):
        weight = torch.load(path)
        
        proto_weight = []
        proto_bias = []
        
        for key in list(weight.keys()):
            if key.startswith('proto_net') and key.endswith('weight'):
                proto_weight.append(weight[key])
            elif key.startswith('proto_net') and key.endswith('bias'):
                proto_bias.append(weight[key])
        
        for i in range(5):
            self.layer[i*2].load([proto_weight[i], proto_bias[i]])
            
    def backward(self, loss, lr):
        
        print('loss :', loss.shape)
        loss = self.layer[8].backward(loss, lr)
        loss = self.layer[7].backward(loss)
        loss = self.layer[6].backward(loss, lr)
        loss = self.interpolate.backward(loss)
        loss = self.layer[5].backward(loss)
        loss = self.layer[4].backward(loss, lr)
        loss = self.layer[3].backward(loss)
        loss = self.layer[2].backward(loss, lr)
        loss = self.layer[1].backward(loss)
        loss = self.layer[0].backward(loss, lr)
        print('\n\nProtonet bw process is over')
        
        return loss

'''