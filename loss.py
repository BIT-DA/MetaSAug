import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb


class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).cuda()

    def update_CV(self, features, labels):
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

        var_temp = features_by_sort - ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(var_temp.permute(1, 2, 0), var_temp.permute(1, 0, 2)).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A))
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(sum_weight_AV + self.Amount.view(C, 1).expand(C, A))
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave - ave_CxA).view(C, A, 1),
                (self.Ave - ave_CxA).view(C, 1, A)
            )
        )

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp.mul(weight_CV)).detach() + additional_CV.detach()
        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()
        self.Amount += onehot.sum(0)


class LDAM_meta(nn.Module):
    def __init__(self, feature_num, class_num, cls_num_list, max_m=0.5, s=30):
        super(LDAM_meta, self).__init__()
        self.estimator = EstimatorCV(feature_num, class_num)
        self.class_num = class_num
        self.cross_entropy = nn.CrossEntropyLoss()

        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s

    def MetaSAug(self, fc, features, y_s, labels_s, s_cv_matrix, ratio,):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.named_leaves())[0][1]

        NxW_ij = weight_m.expand(N, C, A)
        NxW_kj = torch.gather(NxW_ij, 1, labels_s.view(N, 1, 1).expand(N, C, A))

        s_CV_temp = s_cv_matrix[labels_s]

        sigma2 = ratio * torch.bmm(torch.bmm(NxW_ij - NxW_kj, s_CV_temp), (NxW_ij - NxW_kj).permute(0, 2, 1))
        sigma2 = sigma2.mul(torch.eye(C).cuda().expand(N, C, C)).sum(2).view(N, C)

        aug_result = y_s + 0.5 * sigma2
        index = torch.zeros_like(y_s, dtype=torch.uint8)
        index.scatter_(1, labels_s.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        aug_result_m = aug_result - batch_m

        output = torch.where(index, aug_result_m, aug_result)
        return output

    def forward(self, fc, features, y_s, labels, ratio, weights, cv, manner):

        self.estimator.update_CV(features.detach(), labels)
        aug_y = self.MetaSAug(fc, features, y_s, labels, cv, \
                                             ratio)
        if manner == "update":
            self.estimator.update_CV(features.detach(), labels)
            loss = F.cross_entropy(aug_y, labels, weight=weights)
        else:
            loss = F.cross_entropy(aug_y, labels, weight=weights)
        return loss

    def get_cv(self):
        return self.estimator.CoVariance

    def update_cv(self, cv):
        self.estimator.CoVariance = cv


