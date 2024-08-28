from components import *
import torch
import random
import time

class ResBlock:
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        self.path1 = [
            conv_layer(in_channels, out_channels, 3, 3, stride = stride, shift=False, quant=True),
            bn_layer(out_channels),
            relu(),
            conv_layer(out_channels, out_channels, 3, 3, stride = 1, shift=False, quant=True),
            bn_layer(out_channels)
        ]
        self.path2 = shortcut
        self.relu = relu()
    
    def train(self):
        self.path1[1].train()
        self.path1[4].train()
        if self.path2 is not None:
            self.path2[1].train()

    def eval(self):
        self.path1[1].eval()
        self.path1[4].eval()
        if self.path2 is not None:
            self.path2[1].eval()
    
    def block_freeze(self):
        self.path1[1].bn_freeze()
        self.path1[4].bn_freeze()
        

    def forward(self, in_tensor):
        x1 = in_tensor.clone()
        x2 = in_tensor.clone()

        # print('\nbottleneck conv1 input :', x1.sum())
        x1 = self.path1[0].forward(x1) #conv1
        # print('Conv1 output :', x1.sum())
        x1 = self.path1[1].forward(x1) #bn1
        # print('BN1 output :', x1.sum())
        x1 = self.path1[2].forward(x1) #relu1
        # print('Relu1 output :', x1.sum())
        
        x1 = self.path1[3].forward(x1) #conv2
        # print('\nConv2 output :', x1.sum())
        x1 = self.path1[4].forward(x1) #bn2
        # print('BN2 output :', x1.sum())
            
        if self.path2 is not None:
            x2 = self.path2[0].forward(x2) #downsample conv
            x2 = self.path2[1].forward(x2) #downsample bn
        
        self.out_tensor = self.relu.forward(x1+x2)
        # print('RELU2 output :', self.out_tensor.sum())
        # print('\nbackbone.py line 72 exit()')
        # exit()
        return self.out_tensor

    def backward(self, out_diff_tensor, lr, weight_decay, momentum):
        #print('\nself.out_tensor.shape:', self.out_tensor.shape)
        #print('out_diff_tensor.shape :', out_diff_tensor.shape)
        assert self.out_tensor.shape == out_diff_tensor.shape
        
        x1 = self.relu.backward(out_diff_tensor) #RELU2
        x2 = x1.clone().detach() #JUNCTION

        
        x1 = self.path1[4].backward(x1, lr, weight_decay, momentum) #BN2
        x1 = self.path1[3].backward(x1, lr, weight_decay, momentum) #CONV2
        
        x1 = self.path1[2].backward(x1)                             #RELU1
        x1 = self.path1[1].backward(x1, lr, weight_decay, momentum) #BN1
        x1 = self.path1[0].backward(x1, lr, weight_decay, momentum) #CONV1
        
        if self.path2 is not None:
            x2 = self.path2[1].backward(x2, lr, weight_decay, momentum) #SHORTCUT BN
            x2 = self.path2[0].backward(x2, lr, weight_decay, momentum) #SHORTCUT CONV
        
        self.resnet_block_error = x1 + x2
        
        return self.resnet_block_error

    def load(self, conv_num, bn_num): 
        self.path1[0].load([conv_num[0][0]])
        self.path1[1].load(bn_num[:4])
        self.path1[3].load([conv_num[0][1]])
        self.path1[4].load(bn_num[4:])

        if self.path2 is not None:
            self.path2[0].load([conv_num[1][0]])
            self.path2[1].load(conv_num[1][1:])

    def check(self):
        print('\nRELU2')
        self.relu.monitor()
        print('BN2')
        self.path1[4].monitor()
        print('CONV2')
        self.path1[3].monitor()

        print('\nRELU1')
        self.path1[2].monitor()
        print('BN1')
        self.path1[1].monitor()
        print('CONV1')
        self.path1[0].monitor()




class resnet18:
    
    def __init__(self):
        self.pre = [
            conv_layer(3, 64, 3, 3, stride=2, shift=False, pad=1, quant=True),
            bn_layer(64),
            relu(),
            conv_layer(64, 64, 3, 3, stride=1, shift=False, pad=1, quant=True),
            bn_layer(64),
            relu(),
            max_pooling(3,3,2,same=True)
        ]
        self.layer1 = self.stack_ResBlock(64, 64, 2, 1)
        self.layer2 = self.stack_ResBlock(64, 128, 2, 2)
        self.layer3 = self.stack_ResBlock(128, 256, 2, 2)
        self.layer4 = self.stack_ResBlock(256, 512, 2, 2)
        
        self.cnt = 0

    def train(self):
        self.pre[1].train()
        for l in self.layer1:
            l.train()
        for l in self.layer2:
            l.train()
        for l in self.layer3:
            l.train()
        for l in self.layer4: 
            l.train()

    def eval(self):
        #print('\n\n evaluation mode is activated \n\n')
        self.pre[1].eval()
        for l in self.layer1:
            l.eval()
        for l in self.layer2:
            l.eval()
        for l in self.layer3:
            l.eval()
        for l in self.layer4:
            l.eval()
    
    def bn_freeze(self):
        self.pre[1].bn_freeze()
        self.layer1[0].block_freeze()
        self.layer1[1].block_freeze()
        self.layer2[0].block_freeze()
        self.layer2[1].block_freeze()
        self.layer3[0].block_freeze()
        self.layer3[1].block_freeze()
        self.layer4[0].block_freeze()
        self.layer4[1].block_freeze()

    def stack_ResBlock(self, in_channels, out_channels, block_num, stride):
        if stride==2:
            shortcut = [
                conv_layer(in_channels, out_channels, 1, 1, stride=stride, shift=False, pad=0),
                bn_layer(out_channels)
            ]
        else:
            shortcut = None
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride=stride, shortcut=shortcut))

        for _ in range(block_num-1):
            layers.append(ResBlock(out_channels, out_channels))

        return layers

    def forward(self, in_tensor):
        
        x = in_tensor
        outs = []
        i = 0
        '''
        print('RGB Image :', x.reshape(-1).sum())
        if self.cnt > 2 :
            print('backbone.py line 187 exit()')
            exit()
        else :
            self.cnt += 1
        '''
        print('input image :', x.shape)
        for i in range(x.shape[0]):
            print('BATCH',i ,'\t', x[i].sum())

        x = self.pre[0].forward(x) # conv1
        x = self.pre[1].forward(x) # bn1
        x = self.pre[2].forward(x) # relu1

        x = self.pre[3].forward(x) # conv2
        x = self.pre[4].forward(x) # bn2
        x = self.pre[5].forward(x) # relu2
        x = self.pre[6].forward(x) # maxpooling
        
        #Layer 1
        #print('Layer 1 FW start!', x.shape)
        x = self.layer1[0].forward(x)
        x = self.layer1[1].forward(x)
        
        x = self.layer2[0].forward(x)
        x = self.layer2[1].forward(x)
        outs.append(x)
        
        x = self.layer3[0].forward(x)
        x = self.layer3[1].forward(x)
        outs.append(x)
        
        x = self.layer4[0].forward(x)
        x = self.layer4[1].forward(x)
        outs.append(x)

        return outs


        # # torch.save(x, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/IMAGE_INPUT.pth'))
        # x = self.pre[0].forward(x) # conv1
        # torch.save(torch.tensor([self.pre[0].act_scale_mean[-1], 1/self.pre[0].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L0_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(x, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L0_CONV_OUT_DQ_FP32.pth'))
        # # torch.save(self.pre[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L0_CONV_IN_INT8.pth'))
        # # torch.save(torch.tensor([self.pre[0].in_scale[-1], 1/self.pre[0].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L0_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.pre[0].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L0_CONV_OUT_INT32.pth'))
        # # torch.save(self.pre[0].weight, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L0_CONV_WT_INT8.pth'))
        # # torch.save(torch.tensor([self.pre[0].wt_scale, 1/self.pre[0].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L0_CONV_WT_SCALE_FP32.pth'))
        # x = self.pre[1].forward(x) # bn1
        # # torch.save(x, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L1_BN_OUT_FP32.pth'))
        # # torch.save(self.pre[1].gamma, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L1_BN_GAMMA_FP32.pth'))
        # # torch.save(self.pre[1].bias, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L1_BN_BIAS_FP32.pth'))
        # # torch.save(self.pre[1].mean, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L1_BN_MEAN_FP32.pth'))
        # # torch.save(self.pre[1].var, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L1_BN_VAR_FP32.pth'))
        # # torch.save(self.pre[1].std, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L1_BN_STD_FP32.pth'))
        # x = self.pre[2].forward(x) # relu1
        # # torch.save(x, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L2_RELU_OUT_FP32.pth'))
        
        # x = self.pre[3].forward(x) # conv2
        # torch.save(torch.tensor([self.pre[3].act_scale_mean[-1], 1/self.pre[3].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L3_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(self.pre[3].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L3_CONV_IN_INT8.pth'))
        # # torch.save(self.pre[3].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L2_RELU_OUT_Q_INT8.pth'))
        # # torch.save(torch.tensor([self.pre[3].in_scale[-1], 1/self.pre[3].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L3_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.pre[3].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L3_CONV_OUT_INT32.pth'))
        # # torch.save(x, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L3_CONV_OUT_DQ_FP32.pth'))
        # # torch.save(self.pre[3].weight, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L3_CONV_WT_INT8.pth'))
        # # torch.save(torch.tensor([self.pre[3].wt_scale, 1/self.pre[3].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L3_CONV_WT_SCALE_FP32.pth'))
        # x = self.pre[4].forward(x) # bn2
        # # torch.save(x, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L4_BN_OUT_FP32.pth'))
        # # torch.save(self.pre[4].gamma, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L4_BN_GAMMA_FP32.pth'))
        # # torch.save(self.pre[4].bias, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L4_BN_BIAS_FP32.pth'))
        # # torch.save(self.pre[4].mean, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L4_BN_MEAN_FP32.pth'))
        # # torch.save(self.pre[4].var, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L4_BN_VAR_FP32.pth'))
        # # torch.save(self.pre[4].std, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L4_BN_STD_FP32.pth'))
        # x = self.pre[5].forward(x) # relu2
        # # torch.save(x, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L5_RELU_OUT_FP32.pth'))
        # x = self.pre[6].forward(x) # maxpooling
        # # torch.save(x, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L6_PREMAXPOOL_OUT_FP32.pth'))

        # # torch.save(x, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L7_CONV_IN.pth'))
        # x = self.layer1[0].forward(x)
        # torch.save(torch.tensor([self.layer1[0].path1[0].act_scale_mean[-1], 1/self.layer1[0].path1[0].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L7_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(self.layer1[0].path1[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L7_CONV_IN_INT8.pth'))
        # # torch.save(torch.tensor([self.layer1[0].path1[0].in_scale[-1], 1/self.layer1[0].path1[0].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L7_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer1[0].path1[0].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L7_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer1[0].path1[0].weight, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L7_CONV_WT_INT8.pth'))
        # # torch.save(torch.tensor([self.layer1[0].path1[0].wt_scale, 1/self.layer1[0].path1[0].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L7_CONV_WT_SCALE_FP32.pth'))
        # # torch.save(self.layer1[0].path1[0].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L7_CONV_OUT_DQ_FP32.pth'))

        # # torch.save(self.layer1[0].path1[1].gamma, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L8_BN_GAMMA_FP32.pth'))
        # # torch.save(self.layer1[0].path1[1].bias, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L8_BN_BIAS_FP32.pth'))
        # # torch.save(self.layer1[0].path1[1].mean, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L8_BN_MEAN_FP32.pth'))
        # # torch.save(self.layer1[0].path1[1].var, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L8_BN_VAR_FP32.pth'))
        # # torch.save(self.layer1[0].path1[1].std, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L8_BN_STD_FP32.pth'))
        # # torch.save(self.layer1[0].path1[1].out_tensor, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L8_BN_OUT_FP32.pth'))

        # # torch.save(self.layer1[0].path1[2].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L9_RELU_OUT_FP32.pth'))
        
        # # torch.save(self.layer1[0].path1[3].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L10_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer1[0].path1[3].act_scale_mean[-1], 1/self.layer1[0].path1[3].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L10_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(self.layer1[0].path1[3].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L9_RELU_OUT_Q_INT8.pth'))
        # # torch.save(torch.tensor([self.layer1[0].path1[3].in_scale[-1], 1/self.layer1[0].path1[3].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L10_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer1[0].path1[3].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L10_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer1[0].path1[3].weight, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L10_CONV_WT_INT8.pth'))
        # # torch.save(torch.tensor([self.layer1[0].path1[3].wt_scale, 1/self.layer1[0].path1[3].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L10_CONV_WT_SCALE_FP32.pth'))
        # # torch.save(self.layer1[0].path1[3].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L10_CONV_OUT_DQ_FP32.pth'))

        # # torch.save(self.layer1[0].path1[4].gamma, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L11_BN_GAMMA_FP32.pth'))
        # # torch.save(self.layer1[0].path1[4].bias, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L11_BN_BIAS_FP32.pth'))
        # # torch.save(self.layer1[0].path1[4].mean, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L11_BN_MEAN_FP32.pth'))
        # # torch.save(self.layer1[0].path1[4].var, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L11_BN_VAR_FP32.pth'))
        # # torch.save(self.layer1[0].path1[4].std, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L11_BN_STD_FP32.pth'))
        # # torch.save(self.layer1[0].path1[4].out_tensor, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L11_BN_OUT_FP32.pth'))

        # # torch.save(self.layer1[0].relu.out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L13_RELU_OUT_FP32.pth'))


        # # torch.save(x, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L14_CONV_IN.pth'))
        # x = self.layer1[1].forward(x)
        # torch.save(torch.tensor([self.layer1[1].path1[0].act_scale_mean[-1], 1/self.layer1[1].path1[0].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L14_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(self.layer1[1].path1[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L14_CONV_IN_INT8.pth'))
        # # torch.save(self.layer1[1].path1[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L13_RELU_OUT_Q_INT8.pth'))
        # # torch.save(torch.tensor([self.layer1[1].path1[0].in_scale[-1], 1/self.layer1[1].path1[0].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L14_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer1[1].path1[0].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L14_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer1[1].path1[0].weight, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L14_CONV_WT_INT8.pth'))
        # # torch.save(torch.tensor([self.layer1[1].path1[0].wt_scale, 1/self.layer1[1].path1[0].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L14_CONV_WT_SCALE_FP32.pth'))
        # # torch.save(self.layer1[1].path1[0].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L14_CONV_OUT_DQ_FP32.pth'))

        # # torch.save(self.layer1[1].path1[1].gamma, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L15_BN_GAMMA_FP32.pth'))
        # # torch.save(self.layer1[1].path1[1].bias, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L15_BN_BIAS_FP32.pth'))
        # # torch.save(self.layer1[1].path1[1].mean, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L15_BN_MEAN_FP32.pth'))
        # # torch.save(self.layer1[1].path1[1].var, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L15_BN_VAR_FP32.pth'))
        # # torch.save(self.layer1[1].path1[1].std, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L15_BN_STD_FP32.pth'))
        # # torch.save(self.layer1[1].path1[1].out_tensor, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L15_BN_OUT_FP32.pth'))

        # # torch.save(self.layer1[1].path1[2].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L16_RELU_OUT_FP32.pth'))

        # # torch.save(self.layer1[1].path1[3].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L17_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer1[1].path1[3].act_scale_mean[-1], 1/self.layer1[1].path1[3].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L17_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(self.layer1[1].path1[3].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L16_RELU_OUT_Q_INT8.pth'))
        # # torch.save(torch.tensor([self.layer1[1].path1[3].in_scale[-1], 1/self.layer1[1].path1[3].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L17_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer1[1].path1[3].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L17_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer1[1].path1[3].weight, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L17_CONV_WT_INT8.pth'))
        # # torch.save(torch.tensor([self.layer1[1].path1[3].wt_scale, 1/self.layer1[1].path1[3].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L17_CONV_WT_SCALE_FP32.pth'))
        # # torch.save(self.layer1[1].path1[3].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L17_CONV_OUT_DQ_FP32.pth'))

        # # torch.save(self.layer1[1].path1[4].gamma, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L18_BN_GAMMA_FP32.pth'))
        # # torch.save(self.layer1[1].path1[4].bias, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L18_BN_BIAS_FP32.pth'))
        # # torch.save(self.layer1[1].path1[4].mean, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L18_BN_MEAN_FP32.pth'))
        # # torch.save(self.layer1[1].path1[4].var, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L18_BN_VAR_FP32.pth'))
        # # torch.save(self.layer1[1].path1[4].std, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L18_BN_STD_FP32.pth'))
        # # torch.save(self.layer1[1].path1[4].out_tensor, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L18_BN_OUT_FP32.pth'))

        
        # # torch.save(self.layer1[1].relu.out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L20_RELU_OUT_FP32.pth'))
        
        # # # Layer 2
        # # print('Layer 2 FW start!', x.shape)
        # # torch.save(x, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L21_CONV_IN.pth'))
        # x = self.layer2[0].forward(x)
        # # torch.save(self.layer2[0].path1[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L21_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer2[0].path1[0].act_scale_mean[-1], 1/self.layer2[0].path1[0].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L21_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(self.layer2[0].path1[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L20_RELU_OUT_Q_INT8.pth'))
        # # torch.save(torch.tensor([self.layer2[0].path1[0].in_scale[-1], 1/self.layer2[0].path1[0].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L21_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer2[0].path1[0].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L21_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer2[0].path1[0].weight, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L21_CONV_WT_INT8.pth'))
        # # torch.save(torch.tensor([self.layer2[0].path1[0].wt_scale, 1/self.layer2[0].path1[0].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L21_CONV_WT_SCALE_FP32.pth'))
        # # torch.save(self.layer2[0].path1[0].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L21_CONV_OUT_DQ_FP32.pth'))

        # # torch.save(self.layer2[0].path1[1].gamma, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L22_BN_GAMMA_FP32.pth'))
        # # torch.save(self.layer2[0].path1[1].bias, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L22_BN_BIAS_FP32.pth'))
        # # torch.save(self.layer2[0].path1[1].mean, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L22_BN_MEAN_FP32.pth'))
        # # torch.save(self.layer2[0].path1[1].var, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L22_BN_VAR_FP32.pth'))
        # # torch.save(self.layer2[0].path1[1].std, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L22_BN_STD_FP32.pth'))
        # # torch.save(self.layer2[0].path1[1].out_tensor, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L22_BN_OUT_FP32.pth'))

        # # torch.save(self.layer2[0].path1[2].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L23_RELU_OUT_FP32.pth'))

        # # torch.save(self.layer2[0].path1[3].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L24_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer2[0].path1[3].act_scale_mean[-1], 1/self.layer2[0].path1[3].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L24_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(self.layer2[0].path1[3].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L23_RELU_OUT_Q_INT8.pth'))
        # # torch.save(torch.tensor([self.layer2[0].path1[3].in_scale[-1], 1/self.layer2[0].path1[3].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L24_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer2[0].path1[3].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L24_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer2[0].path1[3].weight, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L24_CONV_WT_INT8.pth'))
        # # torch.save(torch.tensor([self.layer2[0].path1[3].wt_scale, 1/self.layer2[0].path1[3].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L24_CONV_WT_SCALE_FP32.pth'))
        # # torch.save(self.layer2[0].path1[3].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L24_CONV_OUT_DQ_FP32.pth'))

        # # torch.save(self.layer2[0].path1[4].gamma, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L25_BN_GAMMA_FP32.pth'))
        # # torch.save(self.layer2[0].path1[4].bias, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L25_BN_BIAS_FP32.pth'))
        # # torch.save(self.layer2[0].path1[4].mean, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L25_BN_MEAN_FP32.pth'))
        # # torch.save(self.layer2[0].path1[4].var, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L25_BN_VAR_FP32.pth'))
        # # torch.save(self.layer2[0].path1[4].std, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L25_BN_STD_FP32.pth'))
        # # torch.save(self.layer2[0].path1[4].out_tensor, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L25_BN_OUT_FP32.pth'))



        # # torch.save(self.layer2[0].path2[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L26_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer2[0].path2[0].act_scale_mean[-1], 1/self.layer2[0].path2[0].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L26_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(torch.tensor([self.layer2[0].path2[0].in_scale[-1], 1/self.layer2[0].path2[0].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L26_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer2[0].path2[0].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L26_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer2[0].path2[0].weight, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L26_CONV_WT_INT8.pth'))
        # # torch.save(torch.tensor([self.layer2[0].path2[0].wt_scale, 1/self.layer2[0].path2[0].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L26_CONV_WT_SCALE_FP32.pth'))
        # # torch.save(self.layer2[0].path2[0].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L26_CONV_OUT_DQ_FP32.pth'))

        # # torch.save(self.layer2[0].path2[1].gamma, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L27_BN_GAMMA_FP32.pth'))
        # # torch.save(self.layer2[0].path2[1].bias, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L27_BN_BIAS_FP32.pth'))
        # # torch.save(self.layer2[0].path2[1].mean, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L27_BN_MEAN_FP32.pth'))
        # # torch.save(self.layer2[0].path2[1].var, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L27_BN_VAR_FP32.pth'))
        # # torch.save(self.layer2[0].path2[1].std, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L27_BN_STD_FP32.pth'))
        # # torch.save(self.layer2[0].path2[1].out_tensor, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L27_BN_OUT_FP32.pth'))

        # # torch.save(self.layer2[0].relu.out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L29_RELU_OUT_FP32.pth'))

        # # torch.save(x, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L30_CONV_IN.pth'))
        # x = self.layer2[1].forward(x)
        # # torch.save(self.layer2[1].path1[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L30_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer2[1].path1[0].act_scale_mean[-1], 1/self.layer2[1].path1[0].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L30_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(self.layer2[1].path1[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L29_RELU_OUT_Q_INT8.pth'))
        # # torch.save(torch.tensor([self.layer2[1].path1[0].in_scale[-1], 1/self.layer2[1].path1[0].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L30_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer2[1].path1[0].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L30_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer2[1].path1[0].weight, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L30_CONV_WT_INT8.pth'))
        # # torch.save(torch.tensor([self.layer2[1].path1[0].wt_scale, 1/self.layer2[1].path1[0].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L30_CONV_WT_SCALE_FP32.pth'))
        # # torch.save(self.layer2[1].path1[0].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L30_CONV_OUT_DQ_FP32.pth'))

        # # torch.save(self.layer2[1].path1[1].gamma, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L31_BN_GAMMA_FP32.pth'))
        # # torch.save(self.layer2[1].path1[1].bias, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L31_BN_BIAS_FP32.pth'))
        # # torch.save(self.layer2[1].path1[1].mean, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L31_BN_MEAN_FP32.pth'))
        # # torch.save(self.layer2[1].path1[1].var, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L31_BN_VAR_FP32.pth'))
        # # torch.save(self.layer2[1].path1[1].std, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L31_BN_STD_FP32.pth'))
        # # torch.save(self.layer2[1].path1[1].out_tensor, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L31_BN_OUT_FP32.pth'))

        # # torch.save(self.layer2[1].path1[2].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L32_RELU_OUT_FP32.pth'))
        # # torch.save(self.layer2[1].path1[3].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L32_RELU_OUT_Q_INT8.pth'))

        # # torch.save(self.layer2[1].path1[3].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L33_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer2[1].path1[3].act_scale_mean[-1], 1/self.layer2[1].path1[3].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L33_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(torch.tensor([self.layer2[1].path1[3].in_scale[-1], 1/self.layer2[1].path1[3].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L33_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer2[1].path1[3].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L33_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer2[1].path1[3].weight, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L33_CONV_WT_INT8.pth'))
        # # torch.save(torch.tensor([self.layer2[1].path1[3].wt_scale, 1/self.layer2[1].path1[3].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L33_CONV_WT_SCALE_FP32.pth'))
        # # torch.save(self.layer2[1].path1[3].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L33_CONV_OUT_DQ_FP32.pth'))

        # # torch.save(self.layer2[1].path1[4].gamma, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L34_BN_GAMMA_FP32.pth'))
        # # torch.save(self.layer2[1].path1[4].bias, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L34_BN_BIAS_FP32.pth'))
        # # torch.save(self.layer2[1].path1[4].mean, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L34_BN_MEAN_FP32.pth'))
        # # torch.save(self.layer2[1].path1[4].var, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L34_BN_VAR_FP32.pth'))
        # # torch.save(self.layer2[1].path1[4].std, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L34_BN_STD_FP32.pth'))
        # # torch.save(self.layer2[1].path1[4].out_tensor, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L34_BN_OUT_FP32.pth'))

        # # torch.save(self.layer2[1].relu.out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L36_RELU_OUT_FP32.pth'))

        # outs.append(x)
        
        # # print('Layer 3 FW start!', x.shape)
        # # torch.save(x, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L37_CONV_IN.pth'))
        # x = self.layer3[0].forward(x)
        # # # torch.save(self.layer3[0].path1[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L37_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer3[0].path1[0].act_scale_mean[-1], 1/self.layer3[0].path1[0].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L37_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(self.layer3[0].path1[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L36_RELU_OUT_Q_INT8.pth'))
        # # torch.save(torch.tensor([self.layer3[0].path1[0].in_scale[-1], 1/self.layer3[0].path1[0].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L37_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer3[0].path1[0].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L37_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer3[0].path1[0].weight, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L37_CONV_WT_INT8.pth'))
        # # torch.save(torch.tensor([self.layer3[0].path1[0].wt_scale, 1/self.layer3[0].path1[0].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L37_CONV_WT_SCALE_FP32.pth'))
        # # torch.save(self.layer3[0].path1[0].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L37_CONV_OUT_DQ_FP32.pth'))

        # # torch.save(self.layer3[0].path1[1].gamma, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L38_BN_GAMMA_FP32.pth'))
        # # torch.save(self.layer3[0].path1[1].bias, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L38_BN_BIAS_FP32.pth'))
        # # torch.save(self.layer3[0].path1[1].mean, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L38_BN_MEAN_FP32.pth'))
        # # torch.save(self.layer3[0].path1[1].var, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L38_BN_VAR_FP32.pth'))
        # # torch.save(self.layer3[0].path1[1].std, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L38_BN_STD_FP32.pth'))
        # # torch.save(self.layer3[0].path1[1].out_tensor, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L38_BN_OUT_FP32.pth'))

        # # torch.save(self.layer3[0].path1[2].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L39_RELU_OUT_FP32.pth'))
        # # torch.save(self.layer3[0].path1[3].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L39_RELU_OUT_Q_INT8.pth'))

        # # torch.save(self.layer3[0].path1[3].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L40_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer3[0].path1[3].act_scale_mean[-1], 1/self.layer3[0].path1[3].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L40_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(torch.tensor([self.layer3[0].path1[3].in_scale[-1], 1/self.layer3[0].path1[3].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L40_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer3[0].path1[3].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L40_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer3[0].path1[3].weight, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L40_CONV_WT_INT8.pth'))
        # # torch.save(torch.tensor([self.layer3[0].path1[3].wt_scale, 1/self.layer3[0].path1[3].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L40_CONV_WT_SCALE_FP32.pth'))
        # # torch.save(self.layer3[0].path1[3].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L40_CONV_OUT_DQ_FP32.pth'))


        # # torch.save(self.layer3[0].path1[4].gamma, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L41_BN_GAMMA_FP32.pth'))
        # # torch.save(self.layer3[0].path1[4].bias, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L41_BN_BIAS_FP32.pth'))
        # # torch.save(self.layer3[0].path1[4].mean, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L41_BN_MEAN_FP32.pth'))
        # # torch.save(self.layer3[0].path1[4].var, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L41_BN_VAR_FP32.pth'))
        # # torch.save(self.layer3[0].path1[4].std, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L41_BN_STD_FP32.pth'))
        # # torch.save(self.layer3[0].path1[4].out_tensor, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L41_BN_OUT_FP32.pth'))



        # # torch.save(self.layer3[0].path2[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L42_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer3[0].path2[0].act_scale_mean[-1], 1/self.layer3[0].path2[0].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L42_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(torch.tensor([self.layer3[0].path2[0].in_scale[-1], 1/self.layer3[0].path2[0].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L42_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer3[0].path2[0].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L42_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer3[0].path2[0].weight, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L42_CONV_WT_INT8.pth'))
        # # torch.save(torch.tensor([self.layer3[0].path2[0].wt_scale, 1/self.layer3[0].path2[0].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L42_CONV_WT_SCALE_FP32.pth'))
        # # torch.save(self.layer3[0].path2[0].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L42_CONV_OUT_DQ_FP32.pth'))

        # # torch.save(self.layer3[0].path2[1].gamma, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L43_BN_GAMMA_FP32.pth'))
        # # torch.save(self.layer3[0].path2[1].bias, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L43_BN_BIAS_FP32.pth'))
        # # torch.save(self.layer3[0].path2[1].mean, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L43_BN_MEAN_FP32.pth'))
        # # torch.save(self.layer3[0].path2[1].var, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L43_BN_VAR_FP32.pth'))
        # # torch.save(self.layer3[0].path2[1].std, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L43_BN_STD_FP32.pth'))
        # # torch.save(self.layer3[0].path2[1].out_tensor, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L43_BN_OUT_FP32.pth'))

        # # torch.save(self.layer3[0].relu.out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L45_RELU_OUT_FP32.pth'))

        # # torch.save(x, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L46_CONV_IN.pth'))
        # x = self.layer3[1].forward(x)
        # # torch.save(self.layer3[1].path1[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L46_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer3[1].path1[0].act_scale_mean[-1], 1/self.layer3[1].path1[0].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L46_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(self.layer3[1].path1[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L45_RELU_OUT_Q_INT8.pth'))
        # # torch.save(torch.tensor([self.layer3[1].path1[0].in_scale[-1], 1/self.layer3[1].path1[0].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L46_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer3[1].path1[0].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L46_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer3[1].path1[0].weight, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L46_CONV_WT_INT8.pth'))
        # # torch.save(torch.tensor([self.layer3[1].path1[0].wt_scale, 1/self.layer3[1].path1[0].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L46_CONV_WT_SCALE_FP32.pth'))
        # # torch.save(self.layer3[1].path1[0].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L46_CONV_OUT_DQ_FP32.pth'))

        # # torch.save(self.layer3[1].path1[1].gamma, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L47_BN_GAMMA_FP32.pth'))
        # # torch.save(self.layer3[1].path1[1].bias, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L47_BN_BIAS_FP32.pth'))
        # # torch.save(self.layer3[1].path1[1].mean, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L47_BN_MEAN_FP32.pth'))
        # # torch.save(self.layer3[1].path1[1].var, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L47_BN_VAR_FP32.pth'))
        # # torch.save(self.layer3[1].path1[1].std, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L47_BN_STD_FP32.pth'))
        # # torch.save(self.layer3[1].path1[1].out_tensor, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L47_BN_OUT_FP32.pth'))

        # # torch.save(self.layer3[1].path1[2].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L48_RELU_OUT_FP32.pth'))
        # # torch.save(self.layer3[1].path1[3].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L48_RELU_OUT_Q_INT8.pth'))

        # # torch.save(self.layer3[1].path1[3].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L49_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer3[1].path1[3].act_scale_mean[-1], 1/self.layer3[1].path1[3].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L49_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(torch.tensor([self.layer3[1].path1[3].in_scale[-1], 1/self.layer3[1].path1[3].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L49_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer3[1].path1[3].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L49_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer3[1].path1[3].weight, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L49_CONV_WT_INT8.pth'))
        # # torch.save(torch.tensor([self.layer3[1].path1[3].wt_scale, 1/self.layer3[1].path1[3].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L49_CONV_WT_SCALE_FP32.pth'))
        # # torch.save(self.layer3[1].path1[3].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L49_CONV_OUT_DQ_FP32.pth'))


        # # torch.save(self.layer3[1].path1[4].gamma, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L50_BN_GAMMA_FP32.pth'))
        # # torch.save(self.layer3[1].path1[4].bias, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L50_BN_BIAS_FP32.pth'))
        # # torch.save(self.layer3[1].path1[4].mean, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L50_BN_MEAN_FP32.pth'))
        # # torch.save(self.layer3[1].path1[4].var, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L50_BN_VAR_FP32.pth'))
        # # torch.save(self.layer3[1].path1[4].std, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L50_BN_STD_FP32.pth'))
        # # torch.save(self.layer3[1].path1[4].out_tensor, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L50_BN_OUT_FP32.pth'))

        # # torch.save(self.layer3[1].relu.out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L52_RELU_OUT_FP32.pth'))

        # outs.append(x)
        
        # # print('Layer 4 FW start!', x.shape)
        # # torch.save(x, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L53_CONV_IN.pth'))
        # x = self.layer4[0].forward(x)
        # # torch.save(self.layer4[0].path1[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L53_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer4[0].path1[0].act_scale_mean[-1], 1/self.layer4[0].path1[0].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L53_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(self.layer4[0].path1[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L52_RELU_OUT_Q_INT8.pth'))
        # # torch.save(torch.tensor([self.layer4[0].path1[0].in_scale[-1], 1/self.layer4[0].path1[0].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L53_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer4[0].path1[0].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L53_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer4[0].path1[0].weight, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L53_CONV_WT_INT8.pth'))
        # # torch.save(torch.tensor([self.layer4[0].path1[0].wt_scale, 1/self.layer4[0].path1[0].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L53_CONV_WT_SCALE_FP32.pth'))
        # # torch.save(self.layer4[0].path1[0].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L53_CONV_OUT_DQ_FP32.pth'))

        # # torch.save(self.layer4[0].path1[1].gamma, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L54_BN_GAMMA_FP32.pth'))
        # # torch.save(self.layer4[0].path1[1].bias, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L54_BN_BIAS_FP32.pth'))
        # # torch.save(self.layer4[0].path1[1].mean, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L54_BN_MEAN_FP32.pth'))
        # # torch.save(self.layer4[0].path1[1].var, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L54_BN_VAR_FP32.pth'))
        # # torch.save(self.layer4[0].path1[1].std, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L54_BN_STD_FP32.pth'))
        # # torch.save(self.layer4[0].path1[1].out_tensor, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L54_BN_OUT_FP32.pth'))

        # # torch.save(self.layer4[0].path1[2].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L55_RELU_OUT_FP32.pth'))
        # # torch.save(self.layer4[0].path1[3].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L55_RELU_OUT_Q_INT8.pth'))

        # # torch.save(self.layer4[0].path1[3].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L56_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer4[0].path1[3].act_scale_mean[-1], 1/self.layer4[0].path1[3].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L56_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(torch.tensor([self.layer4[0].path1[3].in_scale[-1], 1/self.layer4[0].path1[3].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L56_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer4[0].path1[3].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L56_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer4[0].path1[3].weight, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L56_CONV_WT_INT8.pth'))
        # # torch.save(torch.tensor([self.layer4[0].path1[3].wt_scale, 1/self.layer4[0].path1[3].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L56_CONV_WT_SCALE_FP32.pth'))
        # # torch.save(self.layer4[0].path1[3].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L56_CONV_OUT_DQ_FP32.pth'))

        # # torch.save(self.layer4[0].path1[4].gamma, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L57_BN_GAMMA_FP32.pth'))
        # # torch.save(self.layer4[0].path1[4].bias, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L57_BN_BIAS_FP32.pth'))
        # # torch.save(self.layer4[0].path1[4].mean, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L57_BN_MEAN_FP32.pth'))
        # # torch.save(self.layer4[0].path1[4].var, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L57_BN_VAR_FP32.pth'))
        # # torch.save(self.layer4[0].path1[4].std, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L57_BN_STD_FP32.pth'))
        # # torch.save(self.layer4[0].path1[4].out_tensor, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L57_BN_OUT_FP32.pth'))

        # # torch.save(self.layer4[0].path2[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L58_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer4[0].path2[0].act_scale_mean[-1], 1/self.layer4[0].path2[0].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L58_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(torch.tensor([self.layer4[0].path2[0].in_scale[-1], 1/self.layer4[0].path2[0].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L58_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer4[0].path2[0].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L58_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer4[0].path2[0].weight, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L58_CONV_WT_INT8.pth'))
        # # torch.save(torch.tensor([self.layer4[0].path2[0].wt_scale, 1/self.layer4[0].path2[0].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L58_CONV_WT_SCALE_FP32.pth'))
        # # torch.save(self.layer4[0].path2[0].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L58_CONV_OUT_DQ_FP32.pth'))

        # # torch.save(self.layer4[0].path2[1].gamma, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L59_BN_GAMMA_FP32.pth'))
        # # torch.save(self.layer4[0].path2[1].bias, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L59_BN_BIAS_FP32.pth'))
        # # torch.save(self.layer4[0].path2[1].mean, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L59_BN_MEAN_FP32.pth'))
        # # torch.save(self.layer4[0].path2[1].var, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L59_BN_VAR_FP32.pth'))
        # # torch.save(self.layer4[0].path2[1].std, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L59_BN_STD_FP32.pth'))
        # # torch.save(self.layer4[0].path2[1].out_tensor, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L59_BN_OUT_FP32.pth'))

        # # torch.save(self.layer4[0].relu.out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L61_RELU_OUT_FP32.pth'))
        
        # # torch.save(x, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L62_CONV_IN.pth'))
        # x = self.layer4[1].forward(x)
        # # torch.save(self.layer4[1].path1[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L62_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer4[1].path1[0].act_scale_mean[-1],1/self.layer4[1].path1[0].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L62_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(self.layer4[1].path1[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L61_RELU_OUT_Q_INT8.pth'))
        # # torch.save(torch.tensor([self.layer4[1].path1[0].in_scale[-1], 1/self.layer4[1].path1[0].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L62_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer4[1].path1[0].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L62_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer4[1].path1[0].weight, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L62_CONV_WT_INT8.pth'))
        # # torch.save(torch.tensor([self.layer4[1].path1[0].wt_scale, 1/self.layer4[1].path1[0].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L62_CONV_WT_SCALE_FP32.pth'))
        # # torch.save(self.layer4[1].path1[0].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L62_CONV_OUT_DQ_FP32.pth'))

        # # torch.save(self.layer4[1].path1[1].gamma, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L63_BN_GAMMA_FP32.pth'))
        # # torch.save(self.layer4[1].path1[1].bias, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L63_BN_BIAS_FP32.pth'))
        # # torch.save(self.layer4[1].path1[1].mean, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L63_BN_MEAN_FP32.pth'))
        # # torch.save(self.layer4[1].path1[1].var, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L63_BN_VAR_FP32.pth'))
        # # torch.save(self.layer4[1].path1[1].std, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L63_BN_STD_FP32.pth'))
        # # torch.save(self.layer4[1].path1[1].out_tensor, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L63_BN_OUT_FP32.pth'))

        # # torch.save(self.layer4[1].path1[2].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L64_RELU_OUT_FP32.pth'))
        # # torch.save(self.layer4[1].path1[3].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L64_RELU_OUT_Q_INT8.pth'))

        # # torch.save(self.layer4[1].path1[3].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L65_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.layer4[1].path1[3].act_scale_mean[-1], 1/self.layer4[1].path1[3].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L65_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(torch.tensor([self.layer4[1].path1[3].in_scale[-1], 1/self.layer4[1].path1[3].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L65_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.layer4[1].path1[3].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L65_CONV_OUT_INT32.pth'))
        # # torch.save(self.layer4[1].path1[3].weight, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L65_CONV_WT_INT8.pth'))
        # # torch.save(torch.tensor([self.layer4[1].path1[3].wt_scale, 1/self.layer4[1].path1[3].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/scale/L65_CONV_WT_SCALE_FP32.pth'))
        # # torch.save(self.layer4[1].path1[3].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L65_CONV_OUT_DQ_FP32.pth'))

        # # torch.save(self.layer4[1].path1[4].gamma, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L66_BN_GAMMA_FP32.pth'))
        # # torch.save(self.layer4[1].path1[4].bias, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L66_BN_BIAS_FP32.pth'))
        # # torch.save(self.layer4[1].path1[4].mean, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L66_BN_MEAN_FP32.pth'))
        # # torch.save(self.layer4[1].path1[4].var, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L66_BN_VAR_FP32.pth'))
        # # torch.save(self.layer4[1].path1[4].std, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/weight/L66_BN_STD_FP32.pth'))
        # # torch.save(self.layer4[1].path1[4].out_tensor, os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L66_BN_OUT_FP32.pth'))

        # # torch.save(self.layer4[1].relu.out_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/BACKBONE/act/L68_RELU_OUT_FP32.pth'))

        # outs.append(x)

        # return outs

    def backward(self, error, lr, weight_decay, momentum):
        self.fpn_error = error
        self.resnet_error = [[],[],[],[],[]]

        #print('Backbone BW')
        #print('\nResNet Layer 4 Start!')
        self.resnet_error[4] = self.layer4[1].backward(self.fpn_error[2], lr, weight_decay, momentum)
        #print('\nLayer4 Block2 Backward')
        #self.layer4[1].check()
        
        self.resnet_error[4] = self.layer4[0].backward(self.resnet_error[4], lr, weight_decay, momentum)
        #print('\nLayer4 Block1 Backward')
        #self.layer4[0].check()
        
        #print('ResNet Layer 3 Start!')
        self.resnet_error[3] = self.layer3[1].backward(self.fpn_error[1] + self.resnet_error[4], lr, weight_decay, momentum)
        #print('\nLayer3 Block2 Backward')
        #self.layer3[1].check()
        self.resnet_error[3] = self.layer3[0].backward(self.resnet_error[3], lr, weight_decay, momentum)
        #print('\nLayer3 Block1 Backward')
        #self.layer3[0].check()
        
        #print('ResNet Layer 2 Start!')
        self.resnet_error[2] = self.layer2[1].backward(self.fpn_error[0] + self.resnet_error[3], lr, weight_decay, momentum)
        #print('\nLayer2 Block2 Backward')
        #self.layer2[1].check()
        self.resnet_error[2] = self.layer2[0].backward(self.resnet_error[2], lr, weight_decay, momentum)
        #print('\nLayer2 Block1 Backward')
        #self.layer2[0].check()
        
        #print('ResNet Layer 1 Start!')
        self.resnet_error[1] = self.layer1[1].backward(self.resnet_error[2], lr, weight_decay, momentum)
        #print('\nLayer1 Block2 Backward')
        #self.layer1[1].check()
        self.resnet_error[1] = self.layer1[0].backward(self.resnet_error[1], lr, weight_decay, momentum)
        #print('\nLayer1 Block1 Backward')
        #self.layer1[0].check()
        
        #print('ResNet Imgae Layer Start!')
        self.resnet_error[0] = self.pre[6].backward(self.resnet_error[1])                               #Maxpooling layer backward
        self.resnet_error[0] = self.pre[5].backward(self.resnet_error[0])                               #ReLU2 Layer Backward
        self.resnet_error[0] = self.pre[4].backward(self.resnet_error[0], lr, weight_decay, momentum)   #BN2 Layer Backward
        self.resnet_error[0] = self.pre[3].backward(self.resnet_error[0], lr, weight_decay, momentum)   #CONV2 Layer Backward

        self.resnet_error[0] = self.pre[2].backward(self.resnet_error[0])                               #ReLU Layer Backward
        #print('RELU1')
        #self.pre[2].monitor()
        self.resnet_error[0] = self.pre[1].backward(self.resnet_error[0], lr, weight_decay, momentum)   #BN1 Layer Backward
        #print('BN1')
        #self.pre[1].monitor()
        _ = self.pre[0].backward(self.resnet_error[0], lr, weight_decay, momentum)                      #Conv1 Layer Backward
        #print('CONV1')
        #self.pre[0].monitor()


        
    
    def inference(self, in_tensor):
        out_tensor = self.forward(in_tensor).reshape(in_tensor.shape[0], -1)
        return torch.argmax(out_tensor, dim=1)

    def save(self, path):
        conv_num = 0
        bn_num = 0
        
        if os.path.exists(path) == False:
            os.mkdir(path)
            
        conv_num = self.pre[0].save(path, conv_num)
        bn_num = self.pre[1].save(path, bn_num)

        for l in self.layer1:
            conv_num, bn_num = l.save(path, conv_num, bn_num)
        for l in self.layer2:
            conv_num, bn_num = l.save(path, conv_num, bn_num)
        for l in self.layer3:
            conv_num, bn_num = l.save(path, conv_num, bn_num)
        for l in self.layer4:
            conv_num, bn_num = l.save(path, conv_num, bn_num)

        self.fc.save(path)

    def load(self, path):
        weight = torch.load(path)
        pre_conv1 = []
        pre_bn1 = []
        pre_conv2 = []
        pre_bn2 = []
        layer1_conv = []
        layer1_bn = []
        layer2_conv = []
        layer2_bn = []
        layer2_down = []
        layer3_conv = []
        layer3_bn = []
        layer3_down = []
        layer4_conv = []
        layer4_bn = []
        layer4_down = []
        
        for key in list(weight.keys()):
            if key.startswith('backbone.conv1.weight'):
                pre_conv1.append(weight[key])
            elif key.startswith('backbone.bn1') and not key.endswith('num_batches_tracked'):
                pre_bn1.append(weight[key])

            elif key.startswith('backbone.conv2.weight'):
                pre_conv2.append(weight[key])
            elif key.startswith('backbone.bn2') and not key.endswith('num_batches_tracked'):
                pre_bn2.append(weight[key])
            
            elif key.startswith('backbone.layers.0.') and (key.find('conv')>0):
                layer1_conv.append(weight[key])
            elif key.startswith('backbone.layers.0.') and (key.find('bn')>0) and not key.endswith('num_batches_tracked'):
                layer1_bn.append(weight[key])
            
            elif key.startswith('backbone.layers.1.') and (key.find('conv')>0):
                layer2_conv.append(weight[key])
            elif key.startswith('backbone.layers.1.') and (key.find('bn')>0) and not key.endswith('num_batches_tracked'):
                layer2_bn.append(weight[key])
            elif key.startswith('backbone.layers.1.0.downsample') and not key.endswith('num_batches_tracked'):
                layer2_down.append(weight[key])
            
            elif key.startswith('backbone.layers.2.') and (key.find('conv')>0):
                layer3_conv.append(weight[key])
            elif key.startswith('backbone.layers.2.') and (key.find('bn')>0) and not key.endswith('num_batches_tracked'):
                layer3_bn.append(weight[key])
            elif key.startswith('backbone.layers.2.0.downsample') and not key.endswith('num_batches_tracked'):
                layer3_down.append(weight[key])
                
            elif key.startswith('backbone.layers.3.') and (key.find('conv')>0):
                layer4_conv.append(weight[key])
            elif key.startswith('backbone.layers.3.') and (key.find('bn')>0) and not key.endswith('num_batches_tracked'):
                layer4_bn.append(weight[key])
            elif key.startswith('backbone.layers.3.0.downsample') and not key.endswith('num_batches_tracked'):
                layer4_down.append(weight[key])
                
            
                
        self.pre[0].load(pre_conv1)
        self.pre[1].load(pre_bn1)
        
        self.pre[3].load(pre_conv2)
        self.pre[4].load(pre_bn2)

        i=0
        for l in self.layer1:
            l.load([layer1_conv[i*2:(i+1)*2]], layer1_bn[i*8:(i+1)*8])
            i+=1
        i=0
        for l in self.layer2:
            if i == 0:
                l.load([layer2_conv[i*2:(i+1)*2], layer2_down], layer2_bn[i*8:(i+1)*8])
            else:
                l.load([layer2_conv[i*2:(i+1)*2]], layer2_bn[i*8:(i+1)*8])
            i+=1
        i=0
        
        for l in self.layer3:
            if i == 0:
                l.load([layer3_conv[i*2:(i+1)*2], layer3_down], layer3_bn[i*8:(i+1)*8])
            else:
                l.load([layer3_conv[i*2:(i+1)*2]], layer3_bn[i*8:(i+1)*8])
            i+=1
        i=0
        
        for l in self.layer4:
            if i == 0:
                l.load([layer4_conv[i*2:(i+1)*2], layer4_down], layer4_bn[i*8:(i+1)*8])
            else: 
                l.load([layer4_conv[i*2:(i+1)*2]], layer4_bn[i*8:(i+1)*8])
            i+=1


