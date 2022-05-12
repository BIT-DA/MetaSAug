# -*- coding: utf-8 -*

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

def MI(outputs_target):
    batch_size = outputs_target.size(0)
    softmax_outs_t = nn.Softmax(dim=1)(outputs_target)
    avg_softmax_outs_t = torch.sum(softmax_outs_t, dim=0) / float(batch_size)
    log_avg_softmax_outs_t = torch.log(avg_softmax_outs_t)
    item1 = -torch.sum(avg_softmax_outs_t * log_avg_softmax_outs_t)
    item2 = -torch.sum(softmax_outs_t * torch.log(softmax_outs_t)) / float(batch_size)
    return item1, item2

class EstimatorMean():
    def __init__(self, feature_num, class_num):
        super(EstimatorMean, self).__init__()
        self.class_num = class_num
        self.Ave = torch.zeros(class_num, feature_num).cuda()  
        self.Amount = torch.zeros(class_num).cuda()

    def update_Mean(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(N, 1, A).expand(N, C, A)
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A))
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(sum_weight_AV + self.Amount.view(C, 1).expand(C, A))
        weight_AV[weight_AV != weight_AV] = 0

        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()
        self.Amount += onehot.sum(0)

# the estimation of covariance matrix
class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda() 
        self.Amount = torch.zeros(class_num).cuda()

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(N, 1, A).expand(N, C, A)
        # onehot = torch.zeros(N, C).cuda()
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        # var_temp = torch.bmm(var_temp.permute(1, 2, 0), var_temp.permute(1, 0, 2)).div(Amount_CxA.view(C, A, 1).expand(C, A, A))
        var_temp = torch.mul(var_temp.permute(1, 2, 0), var_temp.permute(1, 2, 0)).sum(2).div(Amount_CxA.view(C, A))

        # sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)
        sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        # weight_CV = sum_weight_CV.div(sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A))
        weight_CV = sum_weight_CV.div(sum_weight_CV + self.Amount.view(C, 1).expand(C, A))
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(sum_weight_AV + self.Amount.view(C, 1).expand(C, A))
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.mul(
                (self.Ave - ave_CxA).view(C, A),
                (self.Ave - ave_CxA).view(C, A)
            )
        )
        # (self.Ave - ave_CxA).pow(2)

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp.mul(
            weight_CV)).detach() + additional_CV.detach()
        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()
        self.Amount += onehot.sum(0)

class Loss_meta(nn.Module):
    def __init__(self, feature_num, class_num):
        super(Loss_meta, self).__init__()
        self.source_estimator = EstimatorCV(feature_num, class_num)
        self.class_num = class_num
        self.cross_entropy = nn.CrossEntropyLoss()

    def MetaSAug(self, fc, features, y_s, labels_s, s_cv_matrix, ratio):
        N = features.size(0) 
        C = self.class_num
        A = features.size(1) 
        
        weight_m = fc 

        NxW_ij = weight_m.expand(N, C, A)
        NxW_kj = torch.gather(NxW_ij, 1, labels_s.view(N, 1, 1).expand(N, C, A)) 

        CV_temp = s_cv_matrix[labels_s]
        sigma2 = ratio * (weight_m - NxW_kj).pow(2).mul(CV_temp.view(N, 1, A).expand(N, C, A)).sum(2)
        aug_result = y_s + 0.5 * sigma2
        return aug_result

    def forward(self, fc, features_source, y_s, labels_source, ratio, weights, cv, mode):

        aug_y = self.MetaSAug(fc, features_source, y_s, labels_source, cv, \
                                             ratio)

        if mode == "update":
            self.source_estimator.update_CV(features_source.detach(), labels_source)
            loss = F.cross_entropy(aug_y, labels_source, weight=weights)
        else:
            loss = F.cross_entropy(aug_y, labels_source, weight=weights)
        return loss

    def get_cv(self):
        return self.source_estimator.CoVariance

    def update_cv(self, cv):
        self.source_estimator.CoVariance = cv
