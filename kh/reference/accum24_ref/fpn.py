import numpy as np
import torch
import copy
import torch.nn.functional as F
from components import *
import tracemalloc

class FPN_class:
    def __init__(self):
        self.lat_layer = [
            conv_layer(512, 256, 1, 1, stride = 1, shift=True, pad=0, quant=True),
            conv_layer(256, 256, 1, 1, stride = 1, shift=True, pad=0, quant=True),
            conv_layer(128, 256, 1, 1, stride = 1, shift=True, pad=0, quant=True),
            ]
            
        self.pred_layer = [
            conv_layer(256, 256, 3, 3, stride = 1, shift=True, quant=True),
            relu(),
            conv_layer(256, 256, 3, 3, stride = 1, shift=True, quant=True),
            relu(),
            conv_layer(256, 256, 3, 3, stride = 1, shift=True, quant=True),
            relu(),
            ]
            
        self.downsample_layer = [
            conv_layer(256, 256, 3, 3, stride = 2, shift=True, quant=True),
            conv_layer(256, 256, 3, 3, stride = 2, shift=True, quant=True),
            ]
        #self.interpolate = fpn_interpolate
        self.upsample = upsample('fpn')
            
    def load(self, path):
        weight = torch.load(path)
        fpn_lat_weight = []
        fpn_lat_bias = []
        fpn_pred_weight = []
        fpn_pred_bias = []
        fpn_down_weight = []
        fpn_down_bias = []
        
        for key in list(weight.keys()):
            if key.startswith('fpn.lat') and key.endswith('weight'):
                fpn_lat_weight.append(weight[key])
            elif key.startswith('fpn.lat') and key.endswith('bias'):
                fpn_lat_bias.append(weight[key])
            elif key.startswith('fpn.pred') and key.endswith('weight'):
                fpn_pred_weight.append(weight[key])
            elif key.startswith('fpn.pred') and key.endswith('bias'):
                fpn_pred_bias.append(weight[key])
            elif key.startswith('fpn.down') and key.endswith('weight'):
                fpn_down_weight.append(weight[key])
            elif key.startswith('fpn.down') and key.endswith('bias'):
                fpn_down_bias.append(weight[key])
        
        for i in range(len(self.lat_layer)):
            self.lat_layer[i].load([fpn_lat_weight[i], fpn_lat_bias[i]])
        
        j = int(len(self.pred_layer)/2)
        for i in range(int(len(self.pred_layer)/2)):
            j -= 1
            self.pred_layer[i*2].load([fpn_pred_weight[j], fpn_pred_bias[j]])
            
        
        for i in range(len(self.downsample_layer)):
            self.downsample_layer[i].load([fpn_down_weight[i], fpn_down_bias[i]])
    
    def forward(self, convout):
        out = []
        x = torch.tensor(0)

        #len(convout) : 3
        for _ in range(len(convout)):
            out.append(x)

        #512 -> 256 channel 18x18
        out[2] = self.lat_layer[0].forward(convout[2])
              
        #256 -> 256 channel 35x35
        h = convout[1].shape[2]
        w = convout[1].shape[3]
        tmp_conv1 = self.lat_layer[1].forward(convout[1])
        tmp_up1 = self.upsample.forward(out[2])
        out[1] = tmp_up1 + tmp_conv1
        
        
        #128 -> 256 channel 69x69
        h = convout[0].shape[2]
        w = convout[0].shape[3]
        tmp_conv2 = self.lat_layer[2].forward(convout[0]) 
        tmp_up2 = self.upsample.forward(out[1])
        out[0] = tmp_up2 + tmp_conv2
   
        # This janky second loop is here because TorchScript.
        j = len(convout)
        out[2] = self.pred_layer[4].forward(out[2])  #CONV  18x18
        out[2] = self.pred_layer[5].forward(out[2])  #RELU  18x18
        
        out[1] = self.pred_layer[2].forward(out[1])  #CONV  35x35
        out[1] = self.pred_layer[3].forward(out[1])  #RELU  35x35
        
        out[0] = self.pred_layer[0].forward(out[0])  #CONV  69x69
        out[0] = self.pred_layer[1].forward(out[0])  #RELU  69x69
        
        # In the original paper, this takes care of P6
        down0 = self.downsample_layer[0].forward(out[2])
        out.append(down0)
        
        down1 = self.downsample_layer[1].forward(out[3])
        out.append(down1)

        return out
        
        
        # # torch.save(convout[2], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L69_CONV_IN_FP32.pth'))
        # out[2] = self.lat_layer[0].forward(convout[2])  #CONV  18x18
        # torch.save(torch.tensor([self.lat_layer[0].act_scale_mean[-1], 1/self.lat_layer[0].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L69_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(self.lat_layer[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L68_RELU_OUT_Q_INT8.pth'))
        # # torch.save(self.lat_layer[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L69_CONV_IN_INT8.pth'))
        # # torch.save(torch.tensor([self.lat_layer[0].in_scale[-1], 1/self.lat_layer[0].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/FPN/scale/L69_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.lat_layer[0].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L69_CONV_OUT_INT32.pth'))
        # # torch.save(self.lat_layer[0].conv_dq_out[-1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L69_CONV_OUT_DQ_FP32.pth'))
        # # torch.save(out[2], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L69_CONV_OUT_BIASED_DQ_FP32.pth'))
        # # torch.save(self.lat_layer[0].weight, os.path.join('./origin_data/pth/One_batch/0iter/FPN/weight/L69_CONV_WT_INT8.pth'))
        # # torch.save(self.lat_layer[0].bias, os.path.join('./origin_data/pth/One_batch/0iter/FPN/weight/L69_CONV_BIAS_FP32.pth'))
        # # torch.save(torch.tensor([self.lat_layer[0].wt_scale, 1/self.lat_layer[0].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/FPN/scale/L69_CONV_WT_SCALE_FP32.pth'))
        
        
        # # # 256 -> 256 channel 35x35
        # h = convout[1].shape[2]
        # w = convout[1].shape[3]
        # # # Interpolate
        # # # x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        # # torch.save(convout[1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L71_CONV_IN_256x35.pth'))
        # tmp_conv1 = self.lat_layer[1].forward(convout[1])
        # tmp_up1 = self.upsample.forward(out[2])
        # out[1] = tmp_up1 + tmp_conv1
        # # torch.save(self.lat_layer[1].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L71_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.lat_layer[1].act_scale_mean[-1], 1/self.lat_layer[1].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L71_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(torch.tensor([self.lat_layer[1].in_scale[-1], 1/self.lat_layer[1].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/FPN/scale/L71_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.lat_layer[1].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L71_CONV_OUT_INT32.pth'))
        # # torch.save(tmp_up1, os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L70_UPSAMPLE_OUT_FP32.pth'))
        # # torch.save(self.lat_layer[1].conv_dq_out[-1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L71_CONV_OUT_DQ_FP32.pth'))
        # # torch.save(tmp_conv1, os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L71_CONV_OUT_BIASED_DQ_FP32.pth'))
        # # torch.save(out[1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/LAT.LAYER1_OUT_FP32.pth'))
        # # torch.save(self.lat_layer[1].weight, os.path.join('./origin_data/pth/One_batch/0iter/FPN/weight/L71_CONV_WT_INT8.pth'))
        # # torch.save(self.lat_layer[1].bias, os.path.join('./origin_data/pth/One_batch/0iter/FPN/weight/L71_CONV_BIAS_FP32.pth'))
        # # torch.save(torch.tensor([self.lat_layer[1].wt_scale, 1/self.lat_layer[1].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/FPN/scale/L71_CONV_WT_SCALE_FP32.pth'))
        
        
        # # # 128 -> 256 channel 69x69
        # h = convout[0].shape[2]
        # w = convout[0].shape[3]
        # # # Interpolate
        # # # x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        # # # x = self.interpolate.forward(x)
        # # torch.save(convout[0], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L73_CONV_IN_128x69.pth'))
        # tmp_conv2 = self.lat_layer[2].forward(convout[0]) 
        # tmp_up2 = self.upsample.forward(out[1])
        # out[0] = tmp_up2 + tmp_conv2
        # # torch.save(self.lat_layer[2].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L73_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.lat_layer[2].act_scale_mean[-1], 1/self.lat_layer[2].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L73_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(torch.tensor([self.lat_layer[2].in_scale[-1], 1/self.lat_layer[2].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/FPN/scale/L73_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.lat_layer[2].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L73_CONV_OUT_INT32.pth'))
        # # torch.save(tmp_up2, os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L72_UPSAMPLE_OUT_FP32.pth'))
        # # torch.save(self.lat_layer[2].conv_dq_out[-1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L73_CONV_OUT_DQ_FP32.pth'))
        # # torch.save(tmp_conv2, os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L73_CONV_OUT_BIASED_DQ_FP32.pth'))
        # # torch.save(out[0], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/LAT.LAYER2_OUT_FP32.pth'))
        # # torch.save(self.lat_layer[2].weight, os.path.join('./origin_data/pth/One_batch/0iter/FPN/weight/L73_CONV_WT_INT8.pth'))
        # # torch.save(self.lat_layer[2].bias, os.path.join('./origin_data/pth/One_batch/0iter/FPN/weight/L73_CONV_BIAS_FP32.pth'))
        # # torch.save(torch.tensor([self.lat_layer[2].wt_scale, 1/self.lat_layer[2].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/FPN/scale/L73_CONV_WT_SCALE_FP32.pth'))
   
        # # # This janky second loop is here because TorchScript.
        # j = len(convout)
        # out[2] = self.pred_layer[4].forward(out[2])  #CONV  18x18
        # # torch.save(self.pred_layer[4].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L74_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.pred_layer[4].act_scale_mean[-1], 1/self.pred_layer[4].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L74_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(torch.tensor([self.pred_layer[4].in_scale[-1], 1/self.pred_layer[4].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/FPN/scale/L74_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.pred_layer[4].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L74_CONV_OUT_INT32.pth'))
        # # torch.save(self.pred_layer[4].conv_dq_out[-1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L74_CONV_OUT_DQ_FP32.pth'))
        # # torch.save(out[2], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L74_CONV_OUT_BIASED_DQ_FP32.pth'))
        # # torch.save(self.pred_layer[4].weight, os.path.join('./origin_data/pth/One_batch/0iter/FPN/weight/L74_CONV_WT_INT8.pth'))
        # # torch.save(self.pred_layer[4].bias, os.path.join('./origin_data/pth/One_batch/0iter/FPN/weight/L74_CONV_BIAS_FP32.pth'))
        # # torch.save(torch.tensor([self.pred_layer[4].wt_scale, 1/self.pred_layer[4].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/FPN/scale/L74_CONV_WT_SCALE_FP32.pth'))
        # out[2] = self.pred_layer[5].forward(out[2])  #RELU  18x18
        # # torch.save(out[2], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L75_RELU_OUT_FP32.pth'))
        
        # out[1] = self.pred_layer[2].forward(out[1])  #CONV  35x35
        # # torch.save(self.pred_layer[2].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L75_RELU_OUT_Q_INT8.pth'))
        # # torch.save(self.pred_layer[2].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L76_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.pred_layer[2].act_scale_mean[-1], 1/self.pred_layer[2].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L76_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(torch.tensor([self.pred_layer[2].in_scale[-1], 1/self.pred_layer[2].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/FPN/scale/L76_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.pred_layer[2].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L76_CONV_OUT_INT32.pth'))
        # # torch.save(self.pred_layer[2].conv_dq_out[-1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L76_CONV_OUT_DQ_FP32.pth'))
        # # torch.save(out[1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L76_CONV_OUT_BIASED_DQ_FP32.pth'))
        # # torch.save(self.pred_layer[2].weight, os.path.join('./origin_data/pth/One_batch/0iter/FPN/weight/L76_CONV_WT_INT8.pth'))
        # # torch.save(self.pred_layer[2].bias, os.path.join('./origin_data/pth/One_batch/0iter/FPN/weight/L76_CONV_BIAS_FP32.pth'))
        # # torch.save(torch.tensor([self.pred_layer[2].wt_scale, 1/self.pred_layer[2].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/FPN/scale/L76_CONV_WT_SCALE_FP32.pth'))
        # out[1] = self.pred_layer[3].forward(out[1])  #RELU  35x35
        # # torch.save(out[1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L77_RELU_OUT_FP32.pth'))
        
        # out[0] = self.pred_layer[0].forward(out[0])  #CONV  69x69
        # # torch.save(self.pred_layer[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L77_RELU_OUT_INT8.pth'))
        # # torch.save(self.pred_layer[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L78_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.pred_layer[0].act_scale_mean[-1], 1/self.pred_layer[0].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L78_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(torch.tensor([self.pred_layer[0].in_scale[-1], 1/self.pred_layer[0].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/FPN/scale/L78_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.pred_layer[0].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L78_CONV_OUT_INT32.pth'))
        # # torch.save(self.pred_layer[0].conv_dq_out[-1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L78_CONV_OUT_DQ_FP32.pth'))
        # # torch.save(out[0], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L78_CONV_OUT_BIASED_DQ_FP32.pth'))
        # # torch.save(self.pred_layer[0].weight, os.path.join('./origin_data/pth/One_batch/0iter/FPN/weight/L78_CONV_WT_INT8.pth'))
        # # torch.save(self.pred_layer[0].bias, os.path.join('./origin_data/pth/One_batch/0iter/FPN/weight/L78_CONV_BIAS_FP32.pth'))
        # # torch.save(torch.tensor([self.pred_layer[0].wt_scale, 1/self.pred_layer[0].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/FPN/scale/L78_CONV_WT_SCALE_FP32.pth'))
        # out[0] = self.pred_layer[1].forward(out[0])  #RELU  69x69
        # # torch.save(out[0], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L79_RELU_OUT_FP32.pth'))
        
        # # # In the original paper, this takes care of P6
        # down0 = self.downsample_layer[0].forward(out[2])
        # # torch.save(self.downsample_layer[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L80_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.downsample_layer[0].act_scale_mean[-1], 1/self.downsample_layer[0].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L80_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(torch.tensor([self.downsample_layer[0].in_scale[-1], 1/self.downsample_layer[0].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/FPN/scale/L80_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.downsample_layer[0].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L80_CONV_OUT_INT32.pth'))
        # # torch.save(self.downsample_layer[0].conv_dq_out[-1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L80_CONV_OUT_DQ_FP32.pth'))
        # # torch.save(down0, os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L80_CONV_OUT_BIASED_DQ_FP32.pth'))
        # # torch.save(self.downsample_layer[0].weight, os.path.join('./origin_data/pth/One_batch/0iter/FPN/weight/L80_CONV_WT_INT8.pth'))
        # # torch.save(self.downsample_layer[0].bias, os.path.join('./origin_data/pth/One_batch/0iter/FPN/weight/L80_CONV_BIAS_FP32.pth'))
        # # torch.save(torch.tensor([self.downsample_layer[0].wt_scale, 1/self.downsample_layer[0].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/FPN/scale/L80_CONV_WT_SCALE_FP32.pth'))
        # out.append(down0)
        
        # down1 = self.downsample_layer[1].forward(out[3])
        # # torch.save(self.downsample_layer[1].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L81_CONV_IN_INT8.pth'))
        # torch.save(torch.tensor([self.downsample_layer[1].act_scale_mean[-1], 1/self.downsample_layer[1].act_scale_mean[-1]]), os.path.join('./origin_data/act_scale_mean/L81_CONV_INPUT_SCALE_MEAN.pth'))
        # # torch.save(torch.tensor([self.downsample_layer[1].in_scale[-1], 1/self.downsample_layer[1].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/0iter/FPN/scale/L81_CONV_IN_SCALE_FP32.pth'))
        # # torch.save(self.downsample_layer[1].quant_out, os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L81_CONV_OUT_INT32.pth'))
        # # torch.save(self.downsample_layer[1].conv_dq_out[-1], os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L81_CONV_OUT_DQ_FP32.pth'))
        # # torch.save(down1, os.path.join('./origin_data/pth/One_batch/0iter/FPN/act/L81_CONV_OUT_BIASED_DQ_FP32.pth'))
        # # torch.save(self.downsample_layer[1].weight, os.path.join('./origin_data/pth/One_batch/0iter/FPN/weight/L81_CONV_WT_INT8.pth'))
        # # torch.save(self.downsample_layer[1].bias, os.path.join('./origin_data/pth/One_batch/0iter/FPN/weight/L81_CONV_BIAS_FP32.pth'))
        # # torch.save(torch.tensor([self.downsample_layer[1].wt_scale, 1/self.downsample_layer[1].wt_scale]), os.path.join('./origin_data/pth/One_batch/0iter/FPN/scale/L81_CONV_WT_SCALE_FP32.pth'))
        # out.append(down1)

        # return out
        

    def backward(self, pred_loss, proto_loss, seg_loss, lr, weight_decay, momentum):
        #print('FPN BW Process Start!')
        self.error = [[],[],[]]
        self.pred_error = [[],[],[]]
        self.down_error = [[],[]]

        # #Downsample backward
        self.down_error[1] = pred_loss[1] + self.downsample_layer[1].backward(pred_loss[0], lr, weight_decay, momentum)
        self.down_error[0] = self.downsample_layer[0].backward(self.down_error[1], lr, weight_decay, momentum)

        #Pred layer backward
        self.pred_error[2] = self.pred_layer[5].backward(pred_loss[2] + self.down_error[0])               #RELU  18x18
        self.pred_error[2] = self.pred_layer[4].backward(self.pred_error[2], lr, weight_decay, momentum)  #CONV  18x18
        self.pred_error[1] = self.pred_layer[3].backward(pred_loss[3])                                    #RELU  35x35
        self.pred_error[1] = self.pred_layer[2].backward(self.pred_error[1], lr, weight_decay, momentum)  #CONV  35x35
        self.pred_error[0] = self.pred_layer[1].backward(proto_loss + pred_loss[4] + seg_loss)            #RELU  69x69
        self.pred_error[0] = self.pred_layer[0].backward(self.pred_error[0], lr, weight_decay, momentum)  #CONV  69x69

        #Error aggregation
        self.error[0] = self.pred_error[0]                                              # 69x69
        self.error[1] = self.pred_error[1] + self.upsample.backward(self.error[0])      # 35x35
        self.error[2] = self.pred_error[2] + self.upsample.backward(self.error[1])      # 18x18

        #Lat layer backward
        self.error[0] = self.lat_layer[2].backward(self.error[0], lr, weight_decay, momentum)    #CONV 69x69
        self.error[1] = self.lat_layer[1].backward(self.error[1], lr, weight_decay, momentum)    #CONV 35x35
        self.error[2] = self.lat_layer[0].backward(self.error[2], lr, weight_decay, momentum)    #CONV 18x18

        return self.error
        
        # #Lat layer backward
        # self.error[0] = self.lat_layer[2].backward(self.error[0], lr, weight_decay, momentum)    #CONV 69x69
        # #print('\nself.lat_layer[2]')
        # #self.lat_layer[2].monitor()
        # self.error[1] = self.lat_layer[1].backward(self.error[1], lr, weight_decay, momentum)    #CONV 35x35
        # #print('\nself.lat_layer[1]')
        # #self.lat_layer[1].monitor()
        # self.error[2] = self.lat_layer[0].backward(self.error[2], lr, weight_decay, momentum)    #CONV 18x18

        # #Downsample backward
        # self.down_error[1] = pred_loss[1] + self.downsample_layer[1].backward(pred_loss[0], lr, weight_decay, momentum)
        # #print('\nFPN self.downsample_layer[0] 5x5 -> 9x9')
        # #self.downsample_layer[1].monitor()
        # self.down_error[0] = self.downsample_layer[0].backward(self.down_error[1], lr, weight_decay, momentum)
        # #print('\nFPN self.downsample_layer[1] 9x9 -> 18x18')
        # #self.downsample_layer[0].monitor()

        # #print('\nself.down_error[0]', self.down_error[0].shape, self.down_error[0].abs().sum())

        # #Pred layer backward
        # self.pred_error[2] = self.pred_layer[5].backward(pred_loss[2] + self.down_error[0])               #RELU  18x18
        # self.pred_error[2] = self.pred_layer[4].backward(self.pred_error[2], lr, weight_decay, momentum)  #CONV  18x18
        # #print('\nself.pred_layer[4] 18x18')
        # #self.pred_layer[5].monitor()
        # #self.pred_layer[4].monitor()
        # #pred_error[2] 18 x 18
                 
        # self.pred_error[1] = self.pred_layer[3].backward(pred_loss[3])                                    #RELU  35x35
        # self.pred_error[1] = self.pred_layer[2].backward(self.pred_error[1], lr, weight_decay, momentum)  #CONV  35x35
        # #print('\nself.pred_layer[2] 35x35')
        # #self.pred_layer[3].monitor()
        # #self.pred_layer[2].monitor()
        # #pred_error[1] 35 x 35
        
        # self.pred_error[0] = self.pred_layer[1].backward(proto_loss + pred_loss[4] + seg_loss)                       #RELU  69x69
        # self.pred_error[0] = self.pred_layer[0].backward(self.pred_error[0], lr, weight_decay, momentum)  #CONV  69x69
        # #print('\nself.pred_layer[0] 69x69')
        # #self.pred_layer[1].monitor()
        # #self.pred_layer[0].monitor()
        # #pred_error[0] 69 x 69
        
        
        
        # #Error aggregation
        # #69x69
        # self.error[0] = self.pred_error[0]
        # #35x35
        # self.error[1] = self.pred_error[1] + self.upsample.backward(self.error[0])
        # #print('\nFPN self.upsample[0]')
        # #self.upsample.monitor()
        # #18x18
        # self.error[2] = self.pred_error[2] + self.upsample.backward(self.error[1])
        # #print('\nFPN self.upsample[1]')
        # #self.upsample.monitor()

        
        # #Lat layer backward
        # self.error[0] = self.lat_layer[2].backward(self.error[0], lr, weight_decay, momentum)    #CONV 69x69
        # #print('\nself.lat_layer[2]')
        # #self.lat_layer[2].monitor()
        # self.error[1] = self.lat_layer[1].backward(self.error[1], lr, weight_decay, momentum)    #CONV 35x35
        # #print('\nself.lat_layer[1]')
        # #self.lat_layer[1].monitor()
        # self.error[2] = self.lat_layer[0].backward(self.error[2], lr, weight_decay, momentum)    #CONV 18x18
        # #print('\nself.lat_layer[0]')
        # #self.lat_layer[0].monitor()
        
        # return self.error
        
        
