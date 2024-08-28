import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from layers import Detect

import numpy as np

from backbone import resnet18
from components import *
from components import wt_dequantizer as dequant
from predictionmodule import Prediction_class
from fpn import FPN_class
from protonet import Protonet_class
import torch.nn.functional as F

import numpy as np
import torch

from customconv2d import customConv2d
from customReLU import customReLU
from identify_pass import proto_bypass_act, pred_bypass_act, seg_bypass_act

class Yolact:
    """


    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║   
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║   
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║   
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝ 


    You can set the arguments by changing them in the backbone config object in config.py.

    Parameters (in cfg.backbone):
        - selected_layers: The indices of the conv layers to use for prediction.
        - pred_scales:     A list with len(selected_layers) containing tuples of scales (see PredictionModule)
        - pred_aspect_ratios: A list of lists of aspect ratios with len(selected_layers) (see PredictionModule)
    """

    def __init__(self, priors):
        self.backbone = resnet18()
        self.num_grids = 0
        self.proto_src = 0
        self.priors = priors
        self.img_h = 550
        self.img_w = 550
        self.relu = relu()
        self.softmax = softmax()
        #For evaluation
        self.detect = Detect(21, bkg_label=0, top_k=200, conf_thresh=0.05, nms_thresh=0.5)
        #self.num_girds : 0
        
        self.proto_net = Protonet_class()
        self.fpn = FPN_class()
        self.selected_layers = [0, 1, 2, 3, 4]

        self.prediction_layers = Prediction_class()
        self.num_heads = 5
        self.training = 0
        
        self.num_classes = 21
        self.mask_dim    = 32
        self.num_priors  = 3
        
        self.proto_bypass = proto_bypass_act()
        self.pred_bypass = [pred_bypass_act(), pred_bypass_act(), pred_bypass_act()]
        
        #self.upfeature = [customConv2d(256, 256, 3, stride=1, bias=True), relu()]
        # self.upfeature = [customConv2d(256, 256, 3, stride=1, padding=1, bias=True), relu()]
        # self.bbox_layer = customConv2d(256, self.num_priors * 4, 3, stride=1, padding=1, bias=True)
        # self.conf_layer = customConv2d(256, self.num_priors * self.num_classes, 3, stride=1, padding=1, bias=True)
        # self.mask_layer = customConv2d(256, self.num_priors * self.mask_dim, 3, stride=1, padding=1, bias=True)
        #self.tanh = tanh()
        self.tanh = torch.tanh
        self.timer_cnt = 0
        self.delay = 0

        self.iter_cnt = 0
        
    def train(self, mode=True):
        #super().train(mode)
        #self.backbone.train()
        self.backbone.eval()
        self.training = True
        #self.backbone.bn_freeze()
        
        # self.semantic_seg_conv = customConv2d(256, 20, 1, stride=1, padding=0, bias=True)
        self.ref_seg_conv = conv_layer(256, 20, 1, 1, stride=1, pad=0, shift=True)
        self.seg_bypass = seg_bypass_act()
        #self.semantic_seg_conv = conv_layer(256, 20, 1, 1, stride=1, pad=0, shift=True)
        

    def eval(self):
        self.backbone.eval()
    
    def forward(self, x):
        """ The input should be of size [batch_size, 3, img_h, img_w] """
        #start = time.process_time()
        # print('\n-----------------------------------------------------------')
        # print('Forward Process')
        outs = self.backbone.forward(x)
        outs = self.fpn.forward(outs)
        proto_out = None               
        proto_x = outs[0]
        #print('Proto Net forward')
        proto_out = self.proto_net.forward(proto_x).requires_grad_(True)
        ref_pred_outs = { 'loc': [], 'conf': [], 'mask': [], 'priors': [] }
        for i in range(self.num_heads):
            pred_x = outs[i]
            pred_tmp = self.prediction_layers.forward(pred_x, self.priors[i])
            j = 0
            for k, v in pred_tmp.items():
                if j < 3:
                    ref_pred_outs[k].append(self.pred_bypass[j].forward(v))
                else:
                    ref_pred_outs[k].append(v)
                j += 1

        for k, v in ref_pred_outs.items():
            ref_temp_v = v[0].clone()
            for i in range(1, len(v)):
                ref_temp_v = torch.cat((ref_temp_v, v[i]), dim=-2)
            ref_pred_outs[k] = ref_temp_v

            
                
                
        #print('Prediction Head FW')
        
        # for i in range(self.num_heads):
        #     x = self.upfeature[0].forward(outs[i])
        #     x = self.upfeature[1].forward(x).requires_grad_(True)
            
        # #print('x.shape',x.shape)
        #     bbox = self.bbox_layer.forward(x).permute((0, 2, 3, 1)).contiguous().view(x.shape[0], -1, 4).requires_grad_(True)
        #     conf = self.conf_layer.forward(x).permute((0, 2, 3, 1)).contiguous().view(x.shape[0], -1, self.num_classes).requires_grad_(True)     
        #     mask = self.mask_layer.forward(x).permute((0, 2, 3, 1)).contiguous().view(x.shape[0], -1, self.mask_dim).requires_grad_(True)
        #     mask = self.tanh(mask)
            
        #     preds = { 'loc': bbox.requires_grad_(True), 'conf': conf.requires_grad_(True), 'mask': mask.requires_grad_(True), 'priors': self.priors[i] }
        #     for k, v in preds.items():
        #         pred_outs[k].append(v)

        # for k, v in pred_outs.items():
        #     temp_v = v[0].clone()
        #     for i in range(1, len(v)):
        #         temp_v = torch.cat((temp_v, v[i]), dim=-2)
        #     pred_outs[k] = temp_v
                
        proto_out = self.proto_bypass.forward(proto_out)
        proto_out = self.relu.forward(proto_out)
        # torch.save(proto_out, os.path.join('./origin_data/pth/PROTONET/LAYER10/act/PROTONET_RELU10_OUT.pth'))   
        proto_out = proto_out.permute((0, 2, 3, 1))
        # print('proto_out', proto_out)
        # torch.save(proto_out, os.path.join('./origin_data/pth/PROTONET/LAYER10/act/PROTONET_OUT_PERMUTED.pth'))   
                
        # pred_outs['proto'] = proto_out
        ref_pred_outs['proto'] = proto_out

        #end = time.process_time()
        #self.delay += end - start
        # print('cnt : ', self.timer_cnt)
        self.timer_cnt += 1
        #print("GPU Latency : {:.4f}".format(self.delay), "\tcnt :", self.timer_cnt)

        if self.training:
            #tracemalloc.start()
            seg_out = self.ref_seg_conv.forward(outs[0]).requires_grad_(True)
            ref_pred_outs['segm'] = self.seg_bypass.forward(seg_out)
            # torch.save(outs[0], os.path.join('./origin_data/pth/SEG_LAYER/act/SEG_CONV_IN.pth'))
            # x = self.ref_seg_conv.forward(outs[0])
            # torch.save(x, os.path.join('./origin_data/pth/SEG_LAYER/act/SEG_CONV_OUT.pth'))   
            # torch.save(self.ref_seg_conv.in_tensor[-1], os.path.join('./origin_data/pth/SEG_LAYER/act/SEG_CONV_INPUT_INT8.pth'))
            # torch.save(torch.tensor([self.ref_seg_conv.in_scale[-1], 1/self.ref_seg_conv.in_scale[-1]]), os.path.join('./origin_data/pth/SEG_LAYER/scale/SEG_CONV_INPUT_SCALE.pth'))
            # torch.save(self.ref_seg_conv.quant_out, os.path.join('./origin_data/pth/SEG_LAYER/act/SEG_CONV_OUT_INT32.pth'))
            # torch.save(self.ref_seg_conv.weight, os.path.join('./origin_data/pth/SEG_LAYER/weight/SEG_CONV_OUT_WT.pth'))
            # torch.save(self.ref_seg_conv.bias, os.path.join('./origin_data/pth/SEG_LAYER/weight/SEG_CONV_OUT_BIAS.pth'))
            # torch.save(torch.tensor([self.ref_seg_conv.wt_scale, 1/self.ref_seg_conv.wt_scale]), os.path.join('./origin_data/pth/SEG_LAYER/scale/SEG_CONV_OUT_SCALE.pth'))

            # print('yolact.py line 182 exit()')
            # exit()

            return ref_pred_outs
        
        else:
            #print("\n\npred_outs['conf']", pred_outs['conf'].shape)
            #pred_outs['conf'] = self.softmax.forward(pred_outs['conf'], -1)
            ref_pred_outs['conf'] = self.softmax.forward(ref_pred_outs['conf'], -1)

            _, max_idx = torch.max(ref_pred_outs['conf'], dim=2)
            max_idx = np.array(max_idx.cpu())
            # np.savetxt('./origin_data/inf_0iter_conf_idx.txt', max_idx, fmt="%03d")
            # max_idx_root = './origin_data/inf_0iter_conf_idx.txt'
            with open("./origin_data/inf_0iter_conf_idx.txt", "w") as file:
                for data in max_idx.flatten():
                    formatted_data = f"{data:03d}"
                    file.write(formatted_data + "\n")
            #for k, v in pred_outs.items():
            #    print(k, v.shape)
            
            #detected = self.detect(pred_outs, self)
            
            #Memory free
            self.prediction_layers.upfeature[0].refresh()
            self.prediction_layers.upfeature[1].refresh()
            self.prediction_layers.bbox_layer.refresh()
            self.prediction_layers.conf_layer.refresh()
            self.prediction_layers.mask_layer.refresh()
            self.prediction_layers.tanh.refresh()

            return self.detect(ref_pred_outs, self)
            
    def load_weights(self, path):
        self.path = path
        self.backbone.load(path)
        self.fpn.load(path)
        self.proto_net.load(path)
        self.prediction_layers.load(path)   
        weight = torch.load(path)     
        # for key in list(weight.keys()):
        #     if key.startswith('prediction') and key.endswith('upfeature.0.weight'):
        #         pred_up_weight = dequant(weight[key][0], weight[key][1])
        #     elif key.startswith('prediction') and key.endswith('upfeature.0.bias'):
        #         pred_up_bias = weight[key]
        #     elif key.startswith('prediction') and key.endswith('bbox_layer.weight'):
        #         pred_bbox_weight = dequant(weight[key][0], weight[key][1])
        #     elif key.startswith('prediction') and key.endswith('bbox_layer.bias'):
        #         pred_bbox_bias = weight[key]
        #     elif key.startswith('prediction') and key.endswith('conf_layer.weight'):
        #         pred_conf_weight = dequant(weight[key][0], weight[key][1])
        #     elif key.startswith('prediction') and key.endswith('conf_layer.bias'):
        #         pred_conf_bias = weight[key]
        #     elif key.startswith('prediction') and key.endswith('mask_layer.weight'):
        #         pred_mask_weight = dequant(weight[key][0], weight[key][1])
        #     elif key.startswith('prediction') and key.endswith('mask_layer.bias'):
        #         pred_mask_bias = weight[key]
        
        
        # self.upfeature[0].load([pred_up_weight, pred_up_bias])
        # self.bbox_layer.load([pred_bbox_weight, pred_bbox_bias])
        # self.conf_layer.load([pred_conf_weight, pred_conf_bias])
        # self.mask_layer.load([pred_mask_weight, pred_mask_bias])
        
        if self.training:
            # semantic_weight = dequant(weight['semantic_seg_conv.weight'][0], weight['semantic_seg_conv.weight'][1])
            # self.semantic_seg_conv.load([semantic_weight, weight['semantic_seg_conv.bias']])
            self.ref_seg_conv.load([weight['semantic_seg_conv.weight'], weight['semantic_seg_conv.bias']])
            
        
    def backward(self, lr, weight_decay, momentum):
        print('YOLACT backward')
        #print('Semantic_Seg Backward')
        self.proto_bypass.zero_grad()
        self.pred_bypass[0].zero_grad()
        self.pred_bypass[1].zero_grad()
        self.pred_bypass[2].zero_grad()

        # print('yolact.py line 252 blocked training (because of act scaling factor mean)')

        # self.backbone.pre[0].refresh()
        # self.backbone.pre[2].refresh()
        # self.backbone.pre[3].refresh()
        # self.backbone.pre[5].refresh()
        
        # self.backbone.layer1[0].path1[0].refresh()
        # self.backbone.layer1[0].path1[2].refresh()
        # self.backbone.layer1[0].path1[3].refresh()
        # self.backbone.layer1[0].relu.refresh()
        # self.backbone.layer1[1].path1[0].refresh()
        # self.backbone.layer1[1].path1[2].refresh()
        # self.backbone.layer1[1].path1[3].refresh()
        # self.backbone.layer1[1].relu.refresh()

        # self.backbone.layer2[0].path1[0].refresh()
        # self.backbone.layer2[0].path1[2].refresh()
        # self.backbone.layer2[0].path1[3].refresh()
        # self.backbone.layer2[0].path2[0].refresh()
        # self.backbone.layer2[0].relu.refresh()
        # self.backbone.layer2[1].path1[0].refresh()
        # self.backbone.layer2[1].path1[2].refresh()
        # self.backbone.layer2[1].path1[3].refresh()
        # self.backbone.layer2[1].relu.refresh()

        # self.backbone.layer3[0].path1[0].refresh()
        # self.backbone.layer3[0].path1[2].refresh()
        # self.backbone.layer3[0].path1[3].refresh()
        # self.backbone.layer3[0].path2[0].refresh()
        # self.backbone.layer3[0].relu.refresh()
        # self.backbone.layer3[1].path1[0].refresh()
        # self.backbone.layer3[1].path1[2].refresh()
        # self.backbone.layer3[1].path1[3].refresh()
        # self.backbone.layer3[1].relu.refresh()

        # self.backbone.layer4[0].path1[0].refresh()
        # self.backbone.layer4[0].path1[2].refresh()
        # self.backbone.layer4[0].path1[3].refresh()
        # self.backbone.layer4[0].path2[0].refresh()
        # self.backbone.layer4[0].relu.refresh()
        # self.backbone.layer4[1].path1[0].refresh()
        # self.backbone.layer4[1].path1[2].refresh()
        # self.backbone.layer4[1].path1[3].refresh()
        # self.backbone.layer4[1].relu.refresh()


        # self.fpn.lat_layer[0].refresh()
        # self.fpn.lat_layer[1].refresh()
        # self.fpn.lat_layer[2].refresh()

        # self.fpn.pred_layer[4].refresh()
        # self.fpn.pred_layer[3].refresh()
        # self.fpn.pred_layer[2].refresh()
        # self.fpn.pred_layer[1].refresh()
        # self.fpn.pred_layer[0].refresh()

        # self.fpn.downsample_layer[0].refresh()
        # self.fpn.downsample_layer[1].refresh()


        # self.proto_net.layer[0].refresh()
        # self.proto_net.layer[1].refresh()
        # self.proto_net.layer[2].refresh()
        # self.proto_net.layer[3].refresh()
        # self.proto_net.layer[4].refresh()
        # self.proto_net.layer[5].refresh()
        # self.proto_net.layer[6].refresh()
        # self.proto_net.layer[7].refresh()
        # self.proto_net.layer[8].refresh()
        # self.proto_net.layer[9].refresh()
        # self.relu.refresh()

        # self.prediction_layers.upfeature[0].refresh()
        # self.prediction_layers.upfeature[1].refresh()
        # self.prediction_layers.conf_layer.refresh()
        # self.prediction_layers.bbox_layer.refresh()
        # self.prediction_layers.mask_layer.refresh()
        # self.prediction_layers.tanh.refresh()

        # self.ref_seg_conv.refresh()

        # self.iter_cnt += 1
        # print('iteration :', self.iter_cnt)
        # if self.iter_cnt == 2:
        #     exit()

        seg_loss = torch.load('./ref_loss/seg_loss.pth')
        # print('Seg BW Start!')
        seg_loss = self.ref_seg_conv.backward(seg_loss, lr, weight_decay, momentum)
        # print('Proto BW Start!')
        proto_loss = self.proto_net.backward(lr, weight_decay, momentum)
        # print('Pred BW Start!')
        prediction_loss = self.prediction_layers.backward(lr, weight_decay, momentum)
        # print('FPN BW Start!')
        fpn_loss = self.fpn.backward(prediction_loss, proto_loss, seg_loss, lr, weight_decay, momentum)
        # print('Backbone BW Start!')
        self.backbone.backward(fpn_loss, lr, weight_decay, momentum)
        
        # fp_weight = dequant(self.prediction_layers.upfeature[0].weight, self.prediction_layers.upfeature[0].wt_scale)
        # self.upfeature[0].load([fp_weight, self.prediction_layers.upfeature[0].bias])
        
        # fp_weight = dequant(self.prediction_layers.bbox_layer.weight, self.prediction_layers.bbox_layer.wt_scale)
        # self.bbox_layer.load([fp_weight, self.prediction_layers.bbox_layer.bias])

        # fp_weight = dequant(self.prediction_layers.conf_layer.weight, self.prediction_layers.conf_layer.wt_scale)
        # self.conf_layer.load([fp_weight, self.prediction_layers.conf_layer.bias])
        
        # fp_weight = dequant(self.prediction_layers.mask_layer.weight, self.prediction_layers.mask_layer.wt_scale)
        # self.mask_layer.load([fp_weight, self.prediction_layers.mask_layer.bias])
        
        # fp_weight = dequant(self.ref_seg_conv.weight, self.ref_seg_conv.wt_scale)
        # self.semantic_seg_conv.load([fp_weight, self.ref_seg_conv.bias])

        
        
    def save_weights(self, save_path):
        state_dict = {}
        
        state_dict['backbone.layers.0.0.conv1.weight'] = [self.backbone.layer1[0].path1[0].weight, self.backbone.layer1[0].path1[0].wt_scale]
        state_dict['backbone.layers.0.0.bn1.weight'] = self.backbone.layer1[0].path1[1].gamma
        state_dict['backbone.layers.0.0.bn1.bias'] = self.backbone.layer1[0].path1[1].bias
        state_dict['backbone.layers.0.0.bn1.running_mean'] = self.backbone.layer1[0].path1[1].moving_avg
        state_dict['backbone.layers.0.0.bn1.running_var'] = self.backbone.layer1[0].path1[1].moving_var

        state_dict['backbone.layers.0.0.conv2.weight'] = [self.backbone.layer1[0].path1[3].weight, self.backbone.layer1[0].path1[3].wt_scale]
        state_dict['backbone.layers.0.0.bn2.weight'] = self.backbone.layer1[0].path1[4].gamma
        state_dict['backbone.layers.0.0.bn2.bias'] = self.backbone.layer1[0].path1[4].bias
        state_dict['backbone.layers.0.0.bn2.running_mean'] = self.backbone.layer1[0].path1[4].moving_avg
        state_dict['backbone.layers.0.0.bn2.running_var'] = self.backbone.layer1[0].path1[4].moving_var

        
        state_dict['backbone.layers.0.1.conv1.weight'] = [self.backbone.layer1[1].path1[0].weight, self.backbone.layer1[1].path1[0].wt_scale]
        state_dict['backbone.layers.0.1.bn1.weight'] = self.backbone.layer1[1].path1[1].gamma
        state_dict['backbone.layers.0.1.bn1.bias'] = self.backbone.layer1[1].path1[1].bias
        state_dict['backbone.layers.0.1.bn1.running_mean'] = self.backbone.layer1[1].path1[1].moving_avg
        state_dict['backbone.layers.0.1.bn1.running_var'] = self.backbone.layer1[1].path1[1].moving_var

        state_dict['backbone.layers.0.1.conv2.weight'] = [self.backbone.layer1[1].path1[3].weight, self.backbone.layer1[1].path1[3].wt_scale]
        state_dict['backbone.layers.0.1.bn2.weight'] = self.backbone.layer1[1].path1[4].gamma
        state_dict['backbone.layers.0.1.bn2.bias'] = self.backbone.layer1[1].path1[4].bias
        state_dict['backbone.layers.0.1.bn2.running_mean'] = self.backbone.layer1[1].path1[4].moving_avg
        state_dict['backbone.layers.0.1.bn2.running_var'] = self.backbone.layer1[1].path1[4].moving_var
        
        
        state_dict['backbone.layers.1.0.conv1.weight'] = [self.backbone.layer2[0].path1[0].weight, self.backbone.layer2[0].path1[0].wt_scale]
        state_dict['backbone.layers.1.0.bn1.weight'] = self.backbone.layer2[0].path1[1].gamma
        state_dict['backbone.layers.1.0.bn1.bias'] = self.backbone.layer2[0].path1[1].bias
        state_dict['backbone.layers.1.0.bn1.running_mean'] = self.backbone.layer2[0].path1[1].moving_avg
        state_dict['backbone.layers.1.0.bn1.running_var'] = self.backbone.layer2[0].path1[1].moving_var
        
        state_dict['backbone.layers.1.0.conv2.weight'] = [self.backbone.layer2[0].path1[3].weight, self.backbone.layer2[0].path1[3].wt_scale]
        state_dict['backbone.layers.1.0.bn2.weight'] = self.backbone.layer2[0].path1[4].gamma
        state_dict['backbone.layers.1.0.bn2.bias'] = self.backbone.layer2[0].path1[4].bias
        state_dict['backbone.layers.1.0.bn2.running_mean'] = self.backbone.layer2[0].path1[4].moving_avg
        state_dict['backbone.layers.1.0.bn2.running_var'] = self.backbone.layer2[0].path1[4].moving_var
        
        state_dict['backbone.layers.1.0.downsample.0.weight'] = [self.backbone.layer2[0].path2[0].weight, self.backbone.layer2[0].path2[0].wt_scale]
        state_dict['backbone.layers.1.0.downsample.1.weight'] = self.backbone.layer2[0].path2[1].gamma
        state_dict['backbone.layers.1.0.downsample.1.bias'] = self.backbone.layer2[0].path2[1].bias
        state_dict['backbone.layers.1.0.downsample.1.running_mean'] = self.backbone.layer2[0].path2[1].moving_avg
        state_dict['backbone.layers.1.0.downsample.1.running_var'] = self.backbone.layer2[0].path2[1].moving_var


        state_dict['backbone.layers.1.1.conv1.weight'] = [self.backbone.layer2[1].path1[0].weight, self.backbone.layer2[1].path1[0].wt_scale]
        state_dict['backbone.layers.1.1.bn1.weight'] = self.backbone.layer2[1].path1[1].gamma
        state_dict['backbone.layers.1.1.bn1.bias'] = self.backbone.layer2[1].path1[1].bias
        state_dict['backbone.layers.1.1.bn1.running_mean'] = self.backbone.layer2[1].path1[1].moving_avg
        state_dict['backbone.layers.1.1.bn1.running_var'] = self.backbone.layer2[1].path1[1].moving_var

        state_dict['backbone.layers.1.1.conv2.weight'] = [self.backbone.layer2[1].path1[3].weight, self.backbone.layer2[1].path1[3].wt_scale]
        state_dict['backbone.layers.1.1.bn2.weight'] = self.backbone.layer2[1].path1[4].gamma
        state_dict['backbone.layers.1.1.bn2.bias'] = self.backbone.layer2[1].path1[4].bias
        state_dict['backbone.layers.1.1.bn2.running_mean'] = self.backbone.layer2[1].path1[4].moving_avg
        state_dict['backbone.layers.1.1.bn2.running_var'] = self.backbone.layer2[1].path1[4].moving_var
        

        state_dict['backbone.layers.2.0.conv1.weight'] = [self.backbone.layer3[0].path1[0].weight, self.backbone.layer3[0].path1[0].wt_scale]
        state_dict['backbone.layers.2.0.bn1.weight'] = self.backbone.layer3[0].path1[1].gamma
        state_dict['backbone.layers.2.0.bn1.bias'] = self.backbone.layer3[0].path1[1].bias
        state_dict['backbone.layers.2.0.bn1.running_mean'] = self.backbone.layer3[0].path1[1].moving_avg
        state_dict['backbone.layers.2.0.bn1.running_var'] = self.backbone.layer3[0].path1[1].moving_var

        state_dict['backbone.layers.2.0.conv2.weight'] = [self.backbone.layer3[0].path1[3].weight, self.backbone.layer3[0].path1[3].wt_scale]
        state_dict['backbone.layers.2.0.bn2.weight'] = self.backbone.layer3[0].path1[4].gamma
        state_dict['backbone.layers.2.0.bn2.bias'] = self.backbone.layer3[0].path1[4].bias
        state_dict['backbone.layers.2.0.bn2.running_mean'] = self.backbone.layer3[0].path1[4].moving_avg
        state_dict['backbone.layers.2.0.bn2.running_var'] = self.backbone.layer3[0].path1[4].moving_var
        
        state_dict['backbone.layers.2.0.downsample.0.weight'] = [self.backbone.layer3[0].path2[0].weight, self.backbone.layer3[0].path2[0].wt_scale]
        state_dict['backbone.layers.2.0.downsample.1.weight'] = self.backbone.layer3[0].path2[1].gamma
        state_dict['backbone.layers.2.0.downsample.1.bias'] = self.backbone.layer3[0].path2[1].bias
        state_dict['backbone.layers.2.0.downsample.1.running_mean'] = self.backbone.layer3[0].path2[1].moving_avg
        state_dict['backbone.layers.2.0.downsample.1.running_var'] = self.backbone.layer3[0].path2[1].moving_var
        
    
        state_dict['backbone.layers.2.1.conv1.weight'] = [self.backbone.layer3[1].path1[0].weight, self.backbone.layer3[1].path1[0].wt_scale]
        state_dict['backbone.layers.2.1.bn1.weight'] = self.backbone.layer3[1].path1[1].gamma
        state_dict['backbone.layers.2.1.bn1.bias'] = self.backbone.layer3[1].path1[1].bias
        state_dict['backbone.layers.2.1.bn1.running_mean'] = self.backbone.layer3[1].path1[1].moving_avg
        state_dict['backbone.layers.2.1.bn1.running_var'] = self.backbone.layer3[1].path1[1].moving_var

        state_dict['backbone.layers.2.1.conv2.weight'] = [self.backbone.layer3[1].path1[3].weight, self.backbone.layer3[1].path1[3].wt_scale]
        state_dict['backbone.layers.2.1.bn2.weight'] = self.backbone.layer3[1].path1[4].gamma
        state_dict['backbone.layers.2.1.bn2.bias'] = self.backbone.layer3[1].path1[4].bias
        state_dict['backbone.layers.2.1.bn2.running_mean'] = self.backbone.layer3[1].path1[4].moving_avg
        state_dict['backbone.layers.2.1.bn2.running_var'] = self.backbone.layer3[1].path1[4].moving_var
        
        
        state_dict['backbone.layers.3.0.conv1.weight'] = [self.backbone.layer4[0].path1[0].weight, self.backbone.layer4[0].path1[0].wt_scale]
        state_dict['backbone.layers.3.0.bn1.weight'] = self.backbone.layer4[0].path1[1].gamma
        state_dict['backbone.layers.3.0.bn1.bias'] = self.backbone.layer4[0].path1[1].bias
        state_dict['backbone.layers.3.0.bn1.running_mean'] = self.backbone.layer4[0].path1[1].moving_avg
        state_dict['backbone.layers.3.0.bn1.running_var'] = self.backbone.layer4[0].path1[1].moving_var

        state_dict['backbone.layers.3.0.conv2.weight'] = [self.backbone.layer4[0].path1[3].weight, self.backbone.layer4[0].path1[3].wt_scale]
        state_dict['backbone.layers.3.0.bn2.weight'] = self.backbone.layer4[0].path1[4].gamma
        state_dict['backbone.layers.3.0.bn2.bias'] = self.backbone.layer4[0].path1[4].bias
        state_dict['backbone.layers.3.0.bn2.running_mean'] = self.backbone.layer4[0].path1[4].moving_avg
        state_dict['backbone.layers.3.0.bn2.running_var'] = self.backbone.layer4[0].path1[4].moving_var

        state_dict['backbone.layers.3.0.downsample.0.weight'] = [self.backbone.layer4[0].path2[0].weight, self.backbone.layer4[0].path2[0].wt_scale]
        state_dict['backbone.layers.3.0.downsample.1.weight'] = self.backbone.layer4[0].path2[1].gamma
        state_dict['backbone.layers.3.0.downsample.1.bias'] = self.backbone.layer4[0].path2[1].bias
        state_dict['backbone.layers.3.0.downsample.1.running_mean'] = self.backbone.layer4[0].path2[1].moving_avg
        state_dict['backbone.layers.3.0.downsample.1.running_var'] = self.backbone.layer4[0].path2[1].moving_var
        
        
        state_dict['backbone.layers.3.1.conv1.weight'] = [self.backbone.layer4[1].path1[0].weight, self.backbone.layer4[1].path1[0].wt_scale]
        state_dict['backbone.layers.3.1.bn1.weight'] = self.backbone.layer4[1].path1[1].gamma
        state_dict['backbone.layers.3.1.bn1.bias'] = self.backbone.layer4[1].path1[1].bias
        state_dict['backbone.layers.3.1.bn1.running_mean'] = self.backbone.layer4[1].path1[1].moving_avg
        state_dict['backbone.layers.3.1.bn1.running_var'] = self.backbone.layer4[1].path1[1].moving_var

        state_dict['backbone.layers.3.1.conv2.weight'] = [self.backbone.layer4[1].path1[3].weight, self.backbone.layer4[1].path1[3].wt_scale]
        state_dict['backbone.layers.3.1.bn2.weight'] = self.backbone.layer4[1].path1[4].gamma
        state_dict['backbone.layers.3.1.bn2.bias'] = self.backbone.layer4[1].path1[4].bias
        state_dict['backbone.layers.3.1.bn2.running_mean'] = self.backbone.layer4[1].path1[4].moving_avg
        state_dict['backbone.layers.3.1.bn2.running_var'] = self.backbone.layer4[1].path1[4].moving_var
        
        state_dict['backbone.conv1.weight'] = [self.backbone.pre[0].weight, self.backbone.pre[0].wt_scale]
        state_dict['backbone.bn1.weight'] = self.backbone.pre[1].gamma
        state_dict['backbone.bn1.bias'] = self.backbone.pre[1].bias
        state_dict['backbone.bn1.running_mean'] = self.backbone.pre[1].moving_avg
        state_dict['backbone.bn1.running_var'] = self.backbone.pre[1].moving_var

        state_dict['backbone.conv2.weight'] = [self.backbone.pre[3].weight, self.backbone.pre[3].wt_scale]
        state_dict['backbone.bn2.weight'] = self.backbone.pre[4].gamma
        state_dict['backbone.bn2.bias'] = self.backbone.pre[4].bias
        state_dict['backbone.bn2.running_mean'] = self.backbone.pre[4].moving_avg
        state_dict['backbone.bn2.running_var'] = self.backbone.pre[4].moving_var

        
        state_dict['proto_net.0.weight'] = [self.proto_net.layer[0].weight, self.proto_net.layer[0].wt_scale]
        state_dict['proto_net.0.bias'] = self.proto_net.layer[0].bias
        state_dict['proto_net.2.weight'] = [self.proto_net.layer[2].weight, self.proto_net.layer[2].wt_scale]
        state_dict['proto_net.2.bias'] = self.proto_net.layer[2].bias
        state_dict['proto_net.4.weight'] = [self.proto_net.layer[4].weight, self.proto_net.layer[4].wt_scale]
        state_dict['proto_net.4.bias'] = self.proto_net.layer[4].bias
        state_dict['proto_net.8.weight'] = [self.proto_net.layer[7].weight, self.proto_net.layer[7].wt_scale]
        state_dict['proto_net.8.bias'] = self.proto_net.layer[7].bias
        state_dict['proto_net.10.weight'] = [self.proto_net.layer[9].weight, self.proto_net.layer[9].wt_scale]
        state_dict['proto_net.10.bias'] = self.proto_net.layer[9].bias
        
        #FPN LAT_LAYERS
        state_dict['fpn.lat_layers.0.weight'] = [self.fpn.lat_layer[0].weight, self.fpn.lat_layer[0].wt_scale]
        state_dict['fpn.lat_layers.0.bias'] = self.fpn.lat_layer[0].bias
        state_dict['fpn.lat_layers.1.weight'] = [self.fpn.lat_layer[1].weight, self.fpn.lat_layer[1].wt_scale]
        state_dict['fpn.lat_layers.1.bias'] = self.fpn.lat_layer[1].bias
        state_dict['fpn.lat_layers.2.weight'] = [self.fpn.lat_layer[2].weight, self.fpn.lat_layer[2].wt_scale]
        state_dict['fpn.lat_layers.2.bias'] = self.fpn.lat_layer[2].bias
        
        #FPN PRED LAYERS
        state_dict['fpn.pred_layers.0.weight'] = [self.fpn.pred_layer[4].weight, self.fpn.pred_layer[4].wt_scale]
        state_dict['fpn.pred_layers.0.bias'] = self.fpn.pred_layer[4].bias
        state_dict['fpn.pred_layers.1.weight'] = [self.fpn.pred_layer[2].weight, self.fpn.pred_layer[2].wt_scale]
        state_dict['fpn.pred_layers.1.bias'] = self.fpn.pred_layer[2].bias
        state_dict['fpn.pred_layers.2.weight'] = [self.fpn.pred_layer[0].weight, self.fpn.pred_layer[0].wt_scale]
        state_dict['fpn.pred_layers.2.bias'] = self.fpn.pred_layer[0].bias
        
        #FPN DOWNSAMPLE LAYER
        state_dict['fpn.downsample_layers.0.weight'] = [self.fpn.downsample_layer[0].weight, self.fpn.downsample_layer[0].wt_scale]
        state_dict['fpn.downsample_layers.0.bias'] = self.fpn.downsample_layer[0].bias
        state_dict['fpn.downsample_layers.1.weight'] = [self.fpn.downsample_layer[1].weight, self.fpn.downsample_layer[1].wt_scale]
        state_dict['fpn.downsample_layers.1.bias'] = self.fpn.downsample_layer[1].bias
        
        
        #PREDICTION MODULE
        #UPFEATURE
        state_dict['prediction_layers.0.upfeature.0.weight'] = [self.prediction_layers.upfeature[0].weight, self.prediction_layers.upfeature[0].wt_scale]
        state_dict['prediction_layers.0.upfeature.0.bias'] = self.prediction_layers.upfeature[0].bias
        
        state_dict['prediction_layers.0.bbox_layer.weight'] = [self.prediction_layers.bbox_layer.weight, self.prediction_layers.bbox_layer.wt_scale]
        state_dict['prediction_layers.0.bbox_layer.bias'] = self.prediction_layers.bbox_layer.bias
        
        state_dict['prediction_layers.0.conf_layer.weight'] = [self.prediction_layers.conf_layer.weight, self.prediction_layers.conf_layer.wt_scale]
        state_dict['prediction_layers.0.conf_layer.bias'] = self.prediction_layers.conf_layer.bias
        
        state_dict['prediction_layers.0.mask_layer.weight'] = [self.prediction_layers.mask_layer.weight, self.prediction_layers.mask_layer.wt_scale]
        state_dict['prediction_layers.0.mask_layer.bias'] = self.prediction_layers.mask_layer.bias
        
        #SEMANTIC SEG CONV
        state_dict['semantic_seg_conv.weight'] = [self.ref_seg_conv.weight, self.ref_seg_conv.wt_scale]
        state_dict['semantic_seg_conv.bias'] = self.ref_seg_conv.bias
        
        torch.save(state_dict, save_path)