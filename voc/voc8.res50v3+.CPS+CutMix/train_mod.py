# -*- coding = utf-8 -*-
# @Time = 2022/5/10 10:12
# Author = Chen
# @File = train_mod.py
# -- coding: utf-8 --
from __future__ import division
import os.path as osp
import os
import sys
import time
import argparse
import math
from time import sleep

import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.distributed as dist
import torch.backends.cudnn as cudnn

from config import config
from dataloader import get_train_loader
from network import Network
from dataloader import VOC
from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
# from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from discriminator import s4GAN_discriminator
from loss import CrossEntropy2d
import numpy as np
from skimage import feature as ft
# try:
#     from apex.parallel import DistributedDataParallel, SyncBatchNorm
# except ImportError:
#     raise ImportError(
#         "Please install apex from https://www.github.com/nvidia/apex .")

try:
    from azureml.core import Run
    azure = True
    run = Run.get_context()
except:
    azure = False

parser = argparse.ArgumentParser()

os.environ['MASTER_PORT'] = '169711'

if os.getenv('debug') is not None:
    is_debug = os.environ['debug']
else:
    is_debug = False


'''
For CutMix.
'''
import mask_gen
from custom_collate import SegCollate
mask_generator = mask_gen.BoxMaskGenerator(prop_range=config.cutmix_mask_prop_range, n_boxes=config.cutmix_boxmask_n_boxes,
                                           random_aspect_ratio=not config.cutmix_boxmask_fixed_aspect_ratio,
                                           prop_by_area=not config.cutmix_boxmask_by_size, within_bounds=not config.cutmix_boxmask_outside_bounds,
                                           invert=not config.cutmix_boxmask_no_invert)

add_mask_params_to_batch = mask_gen.AddMaskParamsToBatch(
    mask_generator
)
collate_fn = SegCollate()
mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)
print(parser.parse_args())


def L1_loss(y_true,y_pre):
    return np.sum(np.abs(y_true-y_pre))


def L2_loss(y_true,y_pre):
    return np.sum(np.square(y_true-y_pre))


def loss_calc(pred, label):
    label = Variable(label.long())
    criterion = CrossEntropy2d(ignore_label=255)  # Ignore label ??
    return criterion(pred, label)


def find_good_maps(D_outs, pred_all):
    count = 0
    for i in range(D_outs.size(0)):
        if D_outs[i] > config.threshold_st:
            count += 1

    if count > 0:
        # print('Above ST-Threshold : ', count, '/', config.batch_size)
        pred_sel =torch.Tensor(count, pred_all.size(1), pred_all.size(2), pred_all.size(3))
        label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3))
        num_sel = 0
        for j in range(D_outs.size(0)):
            if D_outs[j] > config.threshold_st:
                pred_sel[num_sel] = pred_all[j]
                label_sel[num_sel] = compute_argmax_map(pred_all[j])
                num_sel += 1
        return pred_sel, label_sel, count
    else:
        return 0, 0, count

def one_hot(label):
    label = label.numpy()
    one_hot = np.zeros((label.shape[0], config.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(config.num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot)
def compute_argmax_map(output):
    output = output.detach().cpu().numpy()
    output = output.transpose((1,2,0))
    output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
    output = torch.from_numpy(output).float()
    return output
with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True

    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader + unsupervised data loader

    train_loader, train_sampler = get_train_loader(engine, VOC, train_source=config.train_source, \
                                                   unsupervised=False, collate_fn=collate_fn)
    unsupervised_train_loader_0, unsupervised_train_sampler_0 = get_train_loader(engine, VOC, \
                train_source=config.unsup_source, unsupervised=True, collate_fn=mask_collate_fn)
    unsupervised_train_loader_1, unsupervised_train_sampler_1 = get_train_loader(engine, VOC, \
                train_source=config.unsup_source, unsupervised=True, collate_fn=collate_fn)

    # if engine.distributed and (engine.local_rank == 0):
    #     tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
    #     generate_tb_dir = config.tb_dir + '/tb'
    #     logger = SummaryWriter(log_dir=tb_dir)
    #     engine.link_tb(tb_dir, generate_tb_dir)

    tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
    generate_tb_dir = config.tb_dir + '/tb'
    logger = SummaryWriter(log_dir = tb_dir)
    engine.link_tb(tb_dir, generate_tb_dir)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
    criterion_csst = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)

    if engine.distributed:
        BatchNorm2d = SyncBatchNorm
    BatchNorm2d = nn.BatchNorm2d

    model = Network(config.num_classes, criterion=criterion,
                    pretrained_model=config.pretrained_model,
                    norm_layer=BatchNorm2d)
    init_weight(model.branch1.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    init_weight(model.branch2.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    # set the lr
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr * engine.world_size

    # define two optimizers
    params_list_l = []
    params_list_l = group_weight(params_list_l, model.branch1.backbone,
                               BatchNorm2d, base_lr)
    for module in model.branch1.business_layer:
        params_list_l = group_weight(params_list_l, module, BatchNorm2d,
                                   base_lr)        # head lr * 10
    # for module in model.branch1.decoder.business_layer:
    #     params_list_l = group_weight(params_list_l, module, BatchNorm2d,
    #                                base_lr)
    # params_list_l saves params  of backbone and head and classifier of branch1
    optimizer_l = torch.optim.SGD(params_list_l,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    params_list_r = []
    params_list_r = group_weight(params_list_r, model.branch2.backbone,
                               BatchNorm2d, base_lr)
    for module in model.branch2.business_layer:
        params_list_r = group_weight(params_list_r, module, BatchNorm2d,
                                   base_lr)        # head lr * 10

    # for module in model.branch2.decoder.business_layer:
    #     params_list_r = group_weight(params_list_r, module, BatchNorm2d,
    #                                  base_lr)

    optimizer_r = torch.optim.SGD(params_list_r,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)



    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    # init D1
    model_D1 = s4GAN_discriminator(num_classes = config.num_classes, dataset = 'pascal_voc')
    # if config.restore_from_D is not None:
    #     model_D1.load_state_dict(torch.load(config.restore_from_D))
    cudnn.benchmark = True
    # optimizer for discriminator network
    optimizer_D1 =torch.optim.Adam(model_D1.parameters(), lr = 1e-4, betas = (0.9, 0.99))

    # init D2
    model_D2 = s4GAN_discriminator(num_classes = config.num_classes, dataset = 'pascal_voc')
    # if config.restore_from_D is not None:
    #     model_D2.load_state_dict(torch.load(config.restore_from_D))
    cudnn.benchmark = True
    # optimizer for discriminator network
    optimizer_D2 = torch.optim.Adam(model_D2.parameters(), lr = 1e-4, betas = (0.9, 0.99))
    criterionD = nn.BCELoss()

    if engine.distributed:
        print('distributed !!')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model)
            model_D1 = torch.nn.DataParallel(model_D1).cuda()
            model_D2 = torch.nn.DataParallel(model_D2).cuda()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = DataParallelModel(model, device_ids=engine.devices)

        model.to(device)
        model_D1.to(device)
        model_D2.to(device)

    engine.register_state(dataloader=train_loader, model=model,model_D1=model_D1,model_D2=model_D2,
                          optimizer_l=optimizer_l, optimizer_r=optimizer_r, optimizer_D1=optimizer_D1, optimizer_D2=optimizer_D2)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    model.train()
    model_D1.train()
    model_D2.train()

    print('begin train')

    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

        if is_debug:
            pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
        else:
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)

        dataloader = iter(train_loader)
        unsupervised_dataloader_0 = iter(unsupervised_train_loader_0)
        unsupervised_dataloader_1 = iter(unsupervised_train_loader_1)

        sum_loss_sup = 0
        sum_loss_sup_r = 0
        sum_cps = 0
        sum_adv1 = 0
        sum_adv2 = 0
        sum_D1 = 0
        sum_D2 = 0
        sum_loss = 0
        sum_fm1 = 0
        sum_fm2 = 0
        sum_st1 = 0
        sum_st2 = 0
        # sum_hog1 = 0
        # sum_hog2 = 0

        ''' supervised part '''
        for idx in pbar:
            optimizer_l.zero_grad()
            optimizer_r.zero_grad()
            optimizer_D1.zero_grad()
            optimizer_D2.zero_grad()
            engine.update_iteration(epoch, idx)
            start_time = time.time()

            # train Segmentation Network
            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False
            for param in model_D2.parameters():
                param.requires_grad = False

            minibatch = dataloader.next()
            unsup_minibatch_0 = unsupervised_dataloader_0.next()
            unsup_minibatch_1 = unsupervised_dataloader_1.next()

            imgs = minibatch['data']
            gts = minibatch['label']
            unsup_imgs_0 = unsup_minibatch_0['data']
            unsup_imgs_1 = unsup_minibatch_1['data']
            mask_params = unsup_minibatch_0['mask_params']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            unsup_imgs_0 = unsup_imgs_0.cuda(non_blocking=True)
            unsup_imgs_1 = unsup_imgs_1.cuda(non_blocking=True)
            mask_params = mask_params.cuda(non_blocking=True)



            # unsupervised loss on model/branch#1
            batch_mix_masks = mask_params
            unsup_imgs_mixed = unsup_imgs_0 * (1 - batch_mix_masks) + unsup_imgs_1 * batch_mix_masks
            # print("-------------")
            # # print(type(unsup_imgs_mixed))
            # print(unsup_imgs_0.shape)
            with torch.no_grad():
                # Estimate the pseudo-label with branch#1 & supervise branch#2
                _, logits_u0_tea_1 = model(unsup_imgs_0, step=1)
                _, logits_u1_tea_1 = model(unsup_imgs_1, step=1)


                logits_u0_tea_1 = logits_u0_tea_1.detach()
                logits_u1_tea_1 = logits_u1_tea_1.detach()
                # Estimate the pseudo-label with branch#2 & supervise branch#1
                _, logits_u0_tea_2 = model(unsup_imgs_0, step=2)
                _, logits_u1_tea_2 = model(unsup_imgs_1, step=2)


                logits_u0_tea_2 = logits_u0_tea_2.detach()
                logits_u1_tea_2 = logits_u1_tea_2.detach()

            # print("!!!!!!!!!!!!!!!")
            # print(unsup_imgs_mixed[0].cpu().numpy().shape)
            # num = []
            # for i in range(0, unsup_imgs_mixed.size(0)):
            #     # print(unsup_imgs_mixed.size(0))
            #     hog_feature, hog_img = ft.hog(np.transpose(unsup_imgs_mixed[0].cpu().numpy(), (1, 2, 0)),
            #                                   orientations = 6,
            #                                   pixels_per_cell = [32, 32], cells_per_block = [2, 2],
            #                                   visualize = True,
            #                                   feature_vector = False)
            #     print(hog_feature.shape)
            #     # 15,15,2,2,6
            #
            #     # print(hog_feature)
            #     d4 = hog_feature.reshape(15,15,24)
            #     print(d4.shape)
            #     d4 = np.transpose(d4, (2, 0, 1))
            #     print(d4.shape)
            #     # (142884,1)
            #     num.append(d4)

            # print(num)
            # print(len(num))
            # if len(num) >= 2:
            #     # print("batchsize>=2")
            #     c = np.concatenate((num[0], num[1]), axis = 0)
            #     for i in range(2, len(num)):
            #         c = np.concatenate((c, num[i]), axis = 0)
            # else:
            #     # print("batchsize=1")
            #     c = num[0]
            #
            # print(c.shape)
            # (428652,1)
            #

            # Mix teacher predictions using same mask
            # It makes no difference whether we do this with logits or probabilities as
            # the mask pixels are either 1 or 0
            logits_cons_tea_1 = logits_u0_tea_1 * (1 - batch_mix_masks) + logits_u1_tea_1 * batch_mix_masks
            _, ps_label_1 = torch.max(logits_cons_tea_1, dim=1)
            ps_label_1 = ps_label_1.long()
            logits_cons_tea_2 = logits_u0_tea_2 * (1 - batch_mix_masks) + logits_u1_tea_2 * batch_mix_masks
            _, ps_label_2 = torch.max(logits_cons_tea_2, dim=1)
            ps_label_2 = ps_label_2.long()

            # Get student#1 prediction for mixed image
            _, logits_cons_stu_1 = model(unsup_imgs_mixed, step=1)
            # Get student#2 prediction for mixed image
            _, logits_cons_stu_2 = model(unsup_imgs_mixed, step=2)
            # print("######")
            # print(logits_cons_stu_2_pred2.shape)
            # print(ps_label_1.shape)
            # print(logits_cons_stu_2.shape)

            # if logits_cons_stu_2_pred2.size(0)>=2:
            #     ljj = np.concatenate((logits_cons_stu_2_pred2[0].cpu().detach().numpy(), logits_cons_stu_2_pred2[1].cpu().detach().numpy()),axis = 0)
            #
            #     for i in range(2, logits_cons_stu_2_pred2.size(0)):
            #         ljj = np.concatenate((ljj, logits_cons_stu_2_pred2[i].cpu().detach().numpy()), axis = 0)
            # else:
            #     ljj = logits_cons_stu_2_pred2.cpu().detach().numpy()
            # print(ljj.shape)
            # # 48,15,15
            # ljj = np.transpose(ljj, (1, 2, 0))
            # # 15,15,48
            # print(ljj.shape)
            # for i in range(0,ljj.shape[0]):
            #     for j in range(0,ljj.shape[1]):
            #         ljj[i][j] = ljj[i][j]/ np.sqrt(np.sum(ljj[i][j] ** 2) + 1e-5 ** 2)
            #         ljj[i][j] = np.minimum(ljj[i][j] , 0.2)
            #         ljj[i][j] = ljj[i][j] / np.sqrt(np.sum(ljj[i][j]  ** 2) + 1e-5 ** 2)
            #
            # ljj = np.transpose(ljj, (2, 0, 1))
            # print(ljj.shape)
            #
            # if logits_cons_stu_1_pred2.size(0)>=2:
            #     ljj2 = np.concatenate((logits_cons_stu_1_pred2[0].cpu().detach().numpy(), logits_cons_stu_1_pred2[1].cpu().detach().numpy()),axis = 0)
            #
            #     for i in range(2, logits_cons_stu_1_pred2.size(0)):
            #         ljj2 = np.concatenate((ljj2, logits_cons_stu_1_pred2[i].cpu().detach().numpy()), axis = 0)
            # else:
            #     ljj2= logits_cons_stu_1_pred2.cpu().detach().numpy()
            # print(ljj2.shape)
            # ljj2 = np.transpose(ljj2, (1, 2, 0))
            #
            # for i in range(0, ljj2.shape[0]):
            #     for j in range(0, ljj2.shape[1]):
            #         ljj2[i][j] = ljj2[i][j] / np.sqrt(np.sum(ljj2[i][j] ** 2) + 1e-5 ** 2)
            #         ljj2[i][j] = np.minimum(ljj2[i][j], 0.2)
            #         ljj2[i][j] = ljj2[i][j] / np.sqrt(np.sum(ljj2[i][j] ** 2) + 1e-5 ** 2)
            # ljj2 = np.transpose(ljj2, (2, 0, 1))
            # print("000000000000000000")
            # # print(c)
            # # print(ljj)
            # loss_hog2 = L1_loss(c , ljj)
            # loss_hog1 = L1_loss(c, ljj2)
            # loss_hog = loss_hog1 + loss_hog2
            # print(loss_hog)

            # dist2 = F.pairwise_distance(logits_cons_stu_2_pred2, torch.tensor(c).cuda(0), p = 2)
            # print(dist2)

            cps_loss = criterion(logits_cons_stu_1, ps_label_2) + criterion(logits_cons_stu_2, ps_label_1)
            # dist.all_reduce(cps_loss, dist.ReduceOp.SUM)
            # cps_loss = cps_loss / engine.world_size
            cps_loss = cps_loss * config.cps_weight








            pred_remain1 = logits_cons_stu_1
            images_remain = unsup_imgs_mixed
            images_remain = (images_remain - torch.min(images_remain)) / (
                        torch.max(images_remain) - torch.min(images_remain))
            pred_cat1 = torch.cat((F.softmax(pred_remain1, dim = 1), images_remain), dim = 1)
            # print("testestestestestest!!!!!!!!!!!!!!!!")
            # print(F.softmax(pred_remain1, dim = 1))
            # confidence and images are D1's input
            D_out_z1, D_out_y_pred1 = model_D1(pred_cat1)  # predicts the D ouput 0-1 and feature map for FM-loss
            # find predicted segmentation maps above threshold
            # print(pred_remain1.shape)
            # [6,21,512,512]
            # print(D_out_z1.shape)
            # [6,1]
            pred_sel1, labels_sel1, count1 = find_good_maps(D_out_z1, pred_remain1)
            # training loss on above threshold segmentation predictions (Cross Entropy Loss)

            # print(pred_sel1.is_cuda)
            # print(labels_sel1.is_cuda)
            if count1 > 0 and epoch > 0:
                loss_st_branch1 = loss_calc(pred_sel1, labels_sel1)
            else:
                loss_st_branch1 = 0.0
            # Concatenates the input images and ground-truth maps for the Districrimator 'Real' input
            D_gt_v = Variable(one_hot(gts.cpu())).cuda()
            # print(D_gt_v)
            # onehot
            images_gt =imgs.cuda()
            images_gt = (images_gt - torch.min(images_gt)) / (torch.max(images_gt) - torch.min(images_gt))

            D_gt_v_cat = torch.cat((D_gt_v, images_gt), dim = 1)
            D_out_z_gt, D_out_y_gt = model_D1(D_gt_v_cat)

            # print(D_out_z_gt)
            # [6,1]confidence
            # L1 loss for Feature Matching Loss
            loss_fm_branch1 = torch.mean(torch.abs(torch.mean(D_out_y_gt, 0) - torch.mean(D_out_y_pred1, 0)))



            pred_remain2 = logits_cons_stu_2
            images_remain = unsup_imgs_mixed
            images_remain = (images_remain - torch.min(images_remain)) / (
                    torch.max(images_remain) - torch.min(images_remain))
            pred_cat2 = torch.cat((F.softmax(pred_remain2, dim = 1), images_remain), dim = 1)
            D_out_z2, D_out_y_pred2 = model_D2(pred_cat2)  # predicts the D ouput 0-1 and feature map for FM-loss
            # find predicted segmentation maps above threshold
            pred_sel2, labels_sel2, count2 = find_good_maps(D_out_z2, pred_remain2)
            # training loss on above threshold segmentation predictions (Cross Entropy Loss)
            if count2 > 0 and epoch > 0:
                loss_st_branch2 = loss_calc(pred_sel2, labels_sel2)
            else:
                loss_st_branch2 = 0.0
            # Concatenates the input images and ground-truth maps for the Districrimator 'Real' input
            D_gt_v = Variable(one_hot(gts.cpu())).cuda()

            images_gt =imgs.cuda()
            images_gt = (images_gt - torch.min(images_gt)) / (torch.max(images_gt) - torch.min(images_gt))

            D_gt_v_cat = torch.cat((D_gt_v, images_gt), dim = 1)
            D_out_z_gt, D_out_y_gt = model_D2(D_gt_v_cat)

            # L1 loss for Feature Matching Loss
            loss_fm_branch2 = torch.mean(torch.abs(torch.mean(D_out_y_gt, 0) - torch.mean(D_out_y_pred2, 0)))


            '''select samples which can fool two discriminators and then self-training to make discriminator weaker'''
            D_out_z12, D_out_y_pred12 = model_D2(pred_cat1)
            pred_sel12, labels_sel12, count12 = find_good_maps(D_out_z12, pred_remain1)
            if count12 > 0 and epoch > 0:
                loss_st2_branch1 = loss_calc(pred_sel12, labels_sel12)
            else:
                loss_st2_branch1 = 0.0

            D_out_z22, D_out_y_pred22 = model_D1(pred_cat2)
            pred_sel22, labels_sel22, count22 = find_good_maps(D_out_z22, pred_remain2)
            if count22 > 0 and epoch > 0:
                loss_st2_branch2 = loss_calc(pred_sel22, labels_sel22)
            else:
                loss_st2_branch2 = 0.0


            # supervised loss on both models
            _, sup_pred_l = model(imgs, step=1)
            _, sup_pred_r = model(imgs, step=2)

            loss_sup = criterion(sup_pred_l, gts)
            # dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
            # loss_sup = loss_sup / engine.world_size

            loss_sup_r = criterion(sup_pred_r, gts)
            # dist.all_reduce(loss_sup_r, dist.ReduceOp.SUM)
            # loss_sup_r = loss_sup_r / engine.world_size
            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            # print(len(optimizer.param_groups))
            optimizer_l.param_groups[0]['lr'] = lr
            optimizer_l.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_l.param_groups)):
                optimizer_l.param_groups[i]['lr'] = lr
            optimizer_r.param_groups[0]['lr'] = lr
            optimizer_r.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_r.param_groups)):
                optimizer_r.param_groups[i]['lr'] = lr

            if count1>0 and count12>0 and epoch>0:
                loss_st_1= loss_st_branch1+ loss_st2_branch1
            elif count1>0 and count12<=0 and epoch>0:
                loss_st_1= loss_st_branch1
            elif count1 <= 0 and count12 > 0 and epoch > 0:
                loss_st_1 = loss_st2_branch1
            else:
                loss_st_1=0.0

            if count2>0 and count22>0 and epoch>0:
                loss_st_2= loss_st_branch2+ loss_st2_branch2
            elif count2>0 and count22<=0 and epoch>0:
                loss_st_2= loss_st_branch2
            elif count2<=0 and count22>0 and epoch>0:
                loss_st_2= loss_st2_branch2
            else:
                loss_st_2=0.0

            loss_adv1 = config.lambda_fm1 * loss_fm_branch1 + config.lambda_st1 * loss_st_1
            loss_adv2 = config.lambda_fm2 * loss_fm_branch2 + config.lambda_st2 * loss_st_2

            # if count1 > 0 and epoch > 0:  # if any good predictions found for self-training loss
            #     loss_adv1 = config.lambda_fm1 * loss_fm_branch1 + config.lambda_st1 * loss_st_branch1
            # else:
            #     loss_adv1 = config.lambda_fm1 * loss_fm_branch1
            #
            # if count2 > 0 and epoch > 0:  # if any good predictions found for self-training loss
            #     loss_adv2 = config.lambda_fm2 * loss_fm_branch2 + config.lambda_st2 * loss_st_branch2
            # else:
            #     loss_adv2 = config.lambda_fm2 * loss_fm_branch2

            loss = loss_sup + loss_sup_r + cps_loss + loss_adv1 + loss_adv2
            # loss = loss_sup + loss_sup_r + cps_loss + loss_adv1 + loss_adv2 +loss_hog
            loss.backward()

            # train D1
            for param in model_D1.parameters():
                param.requires_grad = True

            # train with pred
            pred_cat1 = pred_cat1.detach()  # detach does not allow the graddients to back propagate.

            D_out_z1, _ = model_D1(pred_cat1)

            y_fake_1 = Variable(torch.zeros(D_out_z1.size(0), 1).cuda())
            loss_D_fake1 = criterionD(D_out_z1, y_fake_1)

            # train with gt
            D_out_z_gt1, _ = model_D1(D_gt_v_cat)

            y_real_1 = Variable(torch.ones(D_out_z_gt1.size(0), 1).cuda())
            loss_D_real1 = criterionD(D_out_z_gt1, y_real_1)

            loss_D1 = (loss_D_fake1 + loss_D_real1) / 2.0
            loss_D1.backward()
            # loss_D_value += loss_D.item()

            # train D2
            for param in model_D2.parameters():
                param.requires_grad = True

            # train with pred
            pred_cat2 = pred_cat2.detach()  # detach does not allow the graddients to back propagate.

            D_out_z2, _ = model_D2(pred_cat2)
            y_fake_2 = Variable(torch.zeros(D_out_z2.size(0), 1).cuda())
            loss_D_fake2 = criterionD(D_out_z2, y_fake_2)

            # train with gt
            D_out_z_gt2, _ = model_D2(D_gt_v_cat)
            y_real_2 = Variable(torch.ones(D_out_z_gt2.size(0), 1).cuda())
            loss_D_real2 = criterionD(D_out_z_gt2, y_real_2)

            loss_D2 = (loss_D_fake2 + loss_D_real2) / 2.0
            loss_D2.backward()
            # loss_D_value += loss_D.item()
            optimizer_l.step()
            optimizer_r.step()
            optimizer_D1.step()
            optimizer_D2.step()

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % loss.item() \
                        + ' loss_sup=%.2f' % loss_sup.item() \
                        + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                        + ' loss_cps=%.4f' % cps_loss.item()\
                        + ' loss_adv1=%.4f' % loss_adv1.item() \
                        + ' loss_adv2=%.4f' % loss_adv2.item() \
                        + ' loss_D1=%.4f' % loss_D1.item() \
                        + ' loss_D2=%.4f' % loss_D2.item()

            sum_D2 += loss_D2.item()
            sum_D1 += loss_D1.item()
            sum_loss += loss.item()
            sum_loss_sup += loss_sup.item()
            sum_loss_sup_r += loss_sup_r.item()
            sum_cps += cps_loss.item()
            sum_adv1 += loss_adv1.item()
            sum_adv2 += loss_adv2.item()
            sum_fm1 += loss_fm_branch1.item()
            sum_fm2 += loss_fm_branch2.item()
            sum_st1 += loss_st_branch1
            sum_st2 += loss_st_branch2
            pbar.set_description(print_str, refresh=False)

            end_time = time.time()

        # if engine.distributed and (engine.local_rank == 0):
        logger.add_scalar('train_loss_sup', sum_loss_sup / len(pbar), epoch)
        logger.add_scalar('train_loss_sup_r', sum_loss_sup_r / len(pbar), epoch)
        logger.add_scalar('train_loss_cps', sum_cps / len(pbar), epoch)
        logger.add_scalar('train_loss_adv1', sum_adv1 / len(pbar), epoch)
        logger.add_scalar('train_loss_adv2', sum_adv2 / len(pbar), epoch)
        logger.add_scalar('train_loss', sum_loss / len(pbar), epoch)

        logger.add_scalar('train_loss_fm1', sum_fm1 / len(pbar), epoch)
        logger.add_scalar('train_loss_st1', sum_st1 / len(pbar), epoch)
        logger.add_scalar('train_loss_D1', loss_D1.item(), epoch*len(pbar)+idx)

        logger.add_scalar('train_loss_fm2', sum_fm2 / len(pbar), epoch)
        logger.add_scalar('train_loss_st2', sum_st2 / len(pbar), epoch)
        logger.add_scalar('train_loss_D2', loss_D2.item() , epoch*len(pbar)+idx)


        if azure and engine.local_rank == 0:
            run.log(name='Supervised Training Loss', value=sum_loss_sup / len(pbar))
            run.log(name='Supervised Training Loss right', value=sum_loss_sup_r / len(pbar))
            run.log(name='Supervised Training Loss CPS', value=sum_cps / len(pbar))

        # if (epoch > config.nepochs // 6) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
        engine.save_and_link_checkpoint(config.snapshot_dir,
                                        config.log_dir,
                                        config.log_dir_link)
        # if (epoch > config.nepochs // 6) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
        #     if engine.distributed and (engine.local_rank == 0):
        #         engine.save_and_link_checkpoint(config.snapshot_dir,
        #                                         config.log_dir,
        #                                         config.log_dir_link)
        #     elif not engine.distributed:
        #         engine.save_and_link_checkpoint(config.snapshot_dir,
        #                                         config.log_dir,
        #                                         config.log_dir_link)