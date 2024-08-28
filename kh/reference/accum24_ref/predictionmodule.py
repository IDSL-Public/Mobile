import numpy as np
import torch
import copy

import csv
import pandas as pd
from components import *
from customconv2d import customConv2d
import tracemalloc

'''
idx0_priors = pd.read_csv('./0_priors.csv').values
idx1_priors = pd.read_csv('./1_priors.csv').values
idx2_priors = pd.read_csv('./2_priors.csv').values
idx3_priors = pd.read_csv('./3_priors.csv').values
idx4_priors = pd.read_csv('./4_priors.csv').values
priors = [idx0_priors, idx1_priors, idx2_priors, idx3_priors, idx4_priors]
out = [0,1,2,3,4]
'''


#(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, bias = True, groups=1, device=None, dtype=None)

class Prediction_class:
    def __init__(self):
        self.num_classes = 21
        self.mask_dim    = 32
        self.num_priors  = 3

        self.upfeature = [conv_layer(256, 256, 3, 3, stride=1, shift=True, pred = True, quant=True), relu()]
        self.bbox_layer = conv_layer(256, self.num_priors * 4, 3, 3, stride=1, shift=True, pred = True, quant=True)
        #self.bbox_layer = customConv2d(256, self.num_priors * 4, 3, stride=1, padding=1, bias=True)
        self.conf_layer = conv_layer(256, self.num_priors * self.num_classes, 3, 3, stride=1, shift=True, pred = True, quant=True)
        #self.conf_layer = customConv2d(256, self.num_priors * self.num_classes, 3, stride=1, padding=1, bias=True)
        self.mask_layer = conv_layer(256, self.num_priors * self.mask_dim, 3, 3, stride=1, shift=True, pred = True, quant=True)
        #self.mask_layer = customConv2d(256, self.num_priors * self.mask_dim, 3, stride=1, padding=1, bias=True)
        self.tanh = tanh()
        self.cnt = 0

    def load(self, path):
        weight = torch.load(path)
        
        for key in list(weight.keys()):
            if key.startswith('prediction') and key.endswith('upfeature.0.weight'):
                pred_up_weight = weight[key]
            elif key.startswith('prediction') and key.endswith('upfeature.0.bias'):
                pred_up_bias = weight[key]
            elif key.startswith('prediction') and key.endswith('bbox_layer.weight'):
                pred_bbox_weight = weight[key]
            elif key.startswith('prediction') and key.endswith('bbox_layer.bias'):
                pred_bbox_bias = weight[key]
            elif key.startswith('prediction') and key.endswith('conf_layer.weight'):
                pred_conf_weight = weight[key]
            elif key.startswith('prediction') and key.endswith('conf_layer.bias'):
                pred_conf_bias = weight[key]
            elif key.startswith('prediction') and key.endswith('mask_layer.weight'):
                pred_mask_weight = weight[key]
            elif key.startswith('prediction') and key.endswith('mask_layer.bias'):
                pred_mask_bias = weight[key]
        
        self.upfeature[0].load([pred_up_weight, pred_up_bias])
        self.bbox_layer.load([pred_bbox_weight, pred_bbox_bias])
        self.conf_layer.load([pred_conf_weight, pred_conf_bias])
        self.mask_layer.load([pred_mask_weight, pred_mask_bias])
    
    
    def forward(self, x, priors):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])

        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        # In case we want to use another module's layers)

        # torch.save(x, os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/act/UP_CONV_IN_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))   
        # x = self.upfeature[0].forward_pred(x)
        # # torch.save(self.upfeature[0].in_tensor[-1], os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/act/UP_CONV_IN_INT8_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))
        # # if (self.upfeature[0].pred_cnt - 1) == -1:
        # #     cur_num = 4
        # # else:
        # #     cur_num = self.upfeature[0].pred_cnt - 1
        # # torch.save(torch.tensor([self.upfeature[0].act_scale_mean[cur_num], 1/self.upfeature[0].act_scale_mean[cur_num]]), os.path.join('./origin_data/act_scale_mean/UPFEATURE_CONV_INPUT_SCALE_MEAN_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))
        # torch.save(torch.tensor([self.upfeature[0].in_scale[-1], 1/self.upfeature[0].in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/scale/UP_CONV_IN_SCALE_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))
        # torch.save(self.upfeature[0].quant_out, os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/act/UP_CONV_OUT_INT32_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))
        # torch.save(self.upfeature[0].conv_dq_out[-1], os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/act/UP_CONV_OUT_DQ_FP32_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))
        # torch.save(self.upfeature[0].out_tensor[-1], os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/act/UP_CONV_OUT_BAISED_DQ_FP32_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))   
        # torch.save(self.upfeature[0].weight, os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/weight/UP_CONV_WT_INT8.pth'))
        # torch.save(self.upfeature[0].bias, os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/weight/UP_CONV_BIAS_FP32.pth'))
        # torch.save(torch.tensor([self.upfeature[0].wt_scale, 1/self.upfeature[0].wt_scale]), os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/scale/UP_CONV_WT_SCALE_FP32.pth'))
        # x = self.upfeature[1].forward_pred(x)
        # torch.save(x, os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/act/UP_RELU_OUT_FP32_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth')) 



        # bbox = self.bbox_layer.forward_pred(x)
        # torch.save(self.bbox_layer.in_tensor[-1], os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/act/BBOX_CONV_IN_INT8_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))
        # # torch.save(torch.tensor([self.bbox_layer.act_scale_mean[cur_num], 1/self.bbox_layer.act_scale_mean[cur_num]]), os.path.join('./origin_data/act_scale_mean/BBOXLAYER_CONV_INPUT_SCALE_MEAN_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))
        # torch.save(torch.tensor([self.bbox_layer.in_scale[-1], 1/self.bbox_layer.in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/scale/BBOX_CONV_IN_SCALE_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))
        # torch.save(self.bbox_layer.quant_out, os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/act/BBOX_CONV_OUT_INT32_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))
        # torch.save(self.bbox_layer.conv_dq_out[-1], os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/act/BBOX_CONV_OUT_DQ_FP32_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))
        # torch.save(self.bbox_layer.out_tensor[-1], os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/act/BBOX_CONV_OUT_BAISED_DQ_FP32_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))     
        # torch.save(self.bbox_layer.weight, os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/weight/BBOX_CONV_WT_INT8.pth'))
        # torch.save(self.bbox_layer.bias, os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/weight/BBOX_CONV_BIAS_FP32.pth'))
        # torch.save(torch.tensor([self.bbox_layer.wt_scale, 1/self.bbox_layer.wt_scale]), os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/scale/BBOX_CONV_WT_SCALE_FP32.pth'))

        # conf = self.conf_layer.forward_pred(x)
        # torch.save(self.conf_layer.in_tensor[-1], os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/act/CONF_CONV_IN_INT8_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))
        # # torch.save(torch.tensor([self.conf_layer.act_scale_mean[cur_num], 1/self.conf_layer.act_scale_mean[cur_num]]), os.path.join('./origin_data/act_scale_mean/CONFLAYER_CONV_INPUT_SCALE_MEAN_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))
        # torch.save(torch.tensor([self.conf_layer.in_scale[-1], 1/self.conf_layer.in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/scale/CONF_CONV_IN_SCALE_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))
        # torch.save(self.conf_layer.quant_out, os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/act/CONF_CONV_OUT_INT32_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))
        # torch.save(self.conf_layer.conv_dq_out[-1], os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/act/CONF_CONV_OUT_DQ_FP32_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))
        # torch.save(self.conf_layer.out_tensor[-1], os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/act/CONF_CONV_OUT_BAISED_DQ_FP32_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))   
        # torch.save(self.conf_layer.weight, os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/weight/CONF_CONV_WT_INT8.pth'))
        # torch.save(self.conf_layer.bias, os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/weight/CONF_CONV_BIAS_FP32.pth'))
        # torch.save(torch.tensor([self.conf_layer.wt_scale, 1/self.conf_layer.wt_scale]), os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/scale/CONF_CONV_WT_SCALE_FP32.pth'))

        # mask = self.mask_layer.forward_pred(x)
        # torch.save(self.mask_layer.in_tensor[-1], os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/act/MASK_CONV_IN_INT8_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))
        # # torch.save(torch.tensor([self.mask_layer.act_scale_mean[cur_num], 1/self.mask_layer.act_scale_mean[cur_num]]), os.path.join('./origin_data/act_scale_mean/MASKLAYER_CONV_INPUT_SCALE_MEAN_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))
        # torch.save(torch.tensor([self.mask_layer.in_scale[-1], 1/self.mask_layer.in_scale[-1]]), os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/scale/MASK_CONV_IN_SCALE_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))
        # torch.save(self.mask_layer.quant_out, os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/act/MASK_CONV_OUT_INT32_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))
        # torch.save(self.mask_layer.conv_dq_out[-1], os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/act/MASK_CONV_OUT_DQ_FP32_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))
        # torch.save(self.mask_layer.out_tensor[-1], os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/act/MASK_CONV_OUT_BIASED_DQ_FP32_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))   
        # torch.save(self.mask_layer.weight, os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/weight/MASK_CONV_WT_INT8.pth'))
        # torch.save(self.mask_layer.bias, os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/weight/MASK_CONV_BIAS_FP32.pth'))
        # torch.save(torch.tensor([self.mask_layer.wt_scale, 1/self.mask_layer.wt_scale]), os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/scale/MASK_CONV_WT_SCALE_FP32.pth'))

        # mask = self.tanh.forward(mask)
        # torch.save(mask, os.path.join('./origin_data/pth/One_batch/inference/0iter/PREDLAYER/act/MASK_TANH_OUT_'+str(x.shape[3])+'x'+str(x.shape[2])+'.pth'))   
        
        # bbox = bbox.permute((0, 2, 3, 1)).contiguous().view(x.shape[0], -1, 4)
        # conf = conf.permute((0, 2, 3, 1)).contiguous().view(x.shape[0], -1, self.num_classes)
        # mask = mask.permute((0, 2, 3, 1)).contiguous().view(x.shape[0], -1, self.mask_dim)

        x = self.upfeature[0].forward_pred(x)
        x = self.upfeature[1].forward_pred(x)
        bbox = self.bbox_layer.forward_pred(x).permute((0, 2, 3, 1)).contiguous().view(x.shape[0], -1, 4)
        conf = self.conf_layer.forward_pred(x).permute((0, 2, 3, 1)).contiguous().view(x.shape[0], -1, self.num_classes)
        mask = self.mask_layer.forward_pred(x)
        mask = self.tanh.forward(mask).permute((0, 2, 3, 1)).contiguous().view(x.shape[0], -1, self.mask_dim)

        preds = None
        preds = { 'loc': bbox.requires_grad_(True), 'conf': conf.requires_grad_(True), 'mask': mask.requires_grad_(True), 'priors': priors }

        return preds
        
    def backward(self, lr, weight_decay, momentum):
        #print('Prediction Module BW Process start!')
        bbox_loss = []
        conf_loss = []
        mask_loss = []
        
        loss_shape = [5, 9, 18, 35, 69]
        for i in range(len(loss_shape)):
            bbox_loss.append(torch.load('./ref_loss/'+str(loss_shape[i])+'_bbox_loss.pth').detach())
            conf_loss.append(torch.load('./ref_loss/'+str(loss_shape[i])+'_conf_loss.pth').detach())
            mask_loss.append(torch.load('./ref_loss/'+str(loss_shape[i])+'_mask_loss.pth').detach())
        
        loss_up = []
        temp_b_loss = self.bbox_layer.backward_pred(bbox_loss[0], 4)
        temp_c_loss = self.conf_layer.backward_pred(conf_loss[0], 4)
        temp_m_loss = self.tanh.backward(mask_loss[0], 4)
        temp_m_loss = self.mask_layer.backward_pred(temp_m_loss, 4)
        
        '''
        print('Pred Head bbox')
        self.bbox_layer.monitor()
        print('Pred Head conf')
        self.conf_layer.monitor()
        print('Pred Head mask')
        self.mask_layer.monitor()
        '''
        
        loss_temp = self.upfeature[1].backward_pred(temp_b_loss + temp_c_loss + temp_m_loss, 4) 
        loss_temp = self.upfeature[0].backward_pred(loss_temp, 4)
        #print('Pred -> FPN Loss ', loss_temp.shape, loss_temp.abs().sum())
        loss_up.append(loss_temp)
        
        
        
        temp_b_loss = self.bbox_layer.backward_pred(bbox_loss[1], 3)
        temp_c_loss = self.conf_layer.backward_pred(conf_loss[1], 3)
        temp_m_loss = self.tanh.backward(mask_loss[1], 3)
        temp_m_loss = self.mask_layer.backward_pred(temp_m_loss, 3)

        '''
        print('Pred Head bbox')
        self.bbox_layer.monitor()
        print('Pred Head conf')
        self.conf_layer.monitor()
        print('Pred Head mask')
        self.mask_layer.monitor()
        '''
        
        
        loss_temp = self.upfeature[1].backward_pred(temp_b_loss + temp_c_loss + temp_m_loss, 3) 
        loss_temp = self.upfeature[0].backward_pred(loss_temp, 3)
        #print('Pred -> FPN Loss ', loss_temp.shape, loss_temp.abs().sum())
        loss_up.append(loss_temp)
        
        
        
        temp_b_loss = self.bbox_layer.backward_pred(bbox_loss[2], 2)
        temp_c_loss = self.conf_layer.backward_pred(conf_loss[2], 2)
        temp_m_loss = self.tanh.backward(mask_loss[2], 2)
        temp_m_loss = self.mask_layer.backward_pred(temp_m_loss, 2)

        '''
        print('Pred Head bbox')
        self.bbox_layer.monitor()
        print('Pred Head conf')
        self.conf_layer.monitor()
        print('Pred Head mask')
        self.mask_layer.monitor()
        '''

        
        
        loss_temp = self.upfeature[1].backward_pred(temp_b_loss + temp_c_loss + temp_m_loss, 2) 
        loss_temp = self.upfeature[0].backward_pred(loss_temp, 2)
        #print('Pred -> FPN Loss ', loss_temp.shape, loss_temp.abs().sum())
        loss_up.append(loss_temp)
        
        
        
        temp_b_loss = self.bbox_layer.backward_pred(bbox_loss[3], 1)
        temp_c_loss = self.conf_layer.backward_pred(conf_loss[3], 1)
        temp_m_loss = self.tanh.backward(mask_loss[3], 1)
        temp_m_loss = self.mask_layer.backward_pred(temp_m_loss, 1)

        '''
        print('Pred Head bbox')
        self.bbox_layer.monitor()
        print('Pred Head conf')
        self.conf_layer.monitor()
        print('Pred Head mask')
        self.mask_layer.monitor()
        '''
        
        
        loss_temp = self.upfeature[1].backward_pred(temp_b_loss + temp_c_loss + temp_m_loss, 1) 
        loss_temp = self.upfeature[0].backward_pred(loss_temp, 1)
        #print('Pred -> FPN Loss ', loss_temp.shape, loss_temp.abs().sum())
        loss_up.append(loss_temp)
        
        
        
        temp_b_loss = self.bbox_layer.backward_pred(bbox_loss[4], 0)
        temp_c_loss = self.conf_layer.backward_pred(conf_loss[4], 0)
        temp_m_loss = self.tanh.backward(mask_loss[4], 0)
        temp_m_loss = self.mask_layer.backward_pred(temp_m_loss, 0)

        '''
        print('Pred Head bbox')
        self.bbox_layer.monitor()
        print('Pred Head conf')
        self.conf_layer.monitor()
        print('Pred Head mask')
        self.mask_layer.monitor()
        '''

        
        loss_temp = self.upfeature[1].backward_pred(temp_b_loss + temp_c_loss + temp_m_loss, 0) 
        loss_temp = self.upfeature[0].backward_pred(loss_temp, 0)
        #print('Pred -> FPN Loss ', loss_temp.shape, loss_temp.abs().sum())
        loss_up.append(loss_temp)
        
        #gradient update

        '''
        print('\nBefore step')
        print('bbox_layer :', self.bbox_layer.weight.shape, self.bbox_layer.weight.abs().sum())
        print('conf_layer :', self.conf_layer.weight.shape, self.conf_layer.weight.abs().sum())
        print('mask_layer :', self.mask_layer.weight.shape, self.mask_layer.weight.abs().sum())
        '''


        self.bbox_layer.step(lr, weight_decay, momentum)
        self.conf_layer.step(lr, weight_decay, momentum)
        self.mask_layer.step(lr, weight_decay, momentum)

        '''
        print('\nAfter step')
        print('bbox_layer :', self.bbox_layer.weight.shape, self.bbox_layer.weight.abs().sum())
        print('conf_layer :', self.conf_layer.weight.shape, self.conf_layer.weight.abs().sum())
        print('mask_layer :', self.mask_layer.weight.shape, self.mask_layer.weight.abs().sum())
        '''


        self.upfeature[1].refresh()
        self.upfeature[0].step(lr, weight_decay, momentum)

        self.tanh.refresh()
        
        return loss_up

'''
test_input0 = np.ones((256 * 69 * 69), dtype=np.float32).reshape(1,256,69,69)
test_input1 = np.ones((256 * 35 * 35), dtype=np.float32).reshape(1,256,35,35)
test_input2 = np.ones((256 * 18 * 18), dtype=np.float32).reshape(1,256,18,18)
test_input3 = np.ones((256 * 9 * 9), dtype=np.float32).reshape(1,256,9,9)
test_input4 = np.ones((256 * 5 * 5), dtype=np.float32).reshape(1,256,5,5)
IN_x = [test_input0, test_input1, test_input2, test_input3, test_input4]

pred_layer = Prediction_class()
for i in range(len(priors)):
    out[i] = pred_layer.forward(IN_x[i], priors[i])
    
for i in range(len(out)):
    print("out[",i,"]['conf'].shape", out[i]['conf'].shape)
'''
