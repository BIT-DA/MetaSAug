import os
import time
import argparse
import random
import copy
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
from data_utils import *
from resnet import *
import shutil
import gc

parser = argparse.ArgumentParser(description='Imbalanced Example')
parser.add_argument('--checkpoint_path', default='path.pth.tar', type=str,
                    help='the path of checkpoint')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--imb_factor', default='0.1', type=float)


args = parser.parse_args()
print('checkpoint_path:', args.checkpoint_path)

params = args.checkpoint_path.split('_')
dataset = args.dataset
imb_factor = args.imb_factor


kwargs = {'num_workers': 4, 'pin_memory': False}
use_cuda = torch.cuda.is_available()

torch.manual_seed(42)

print('start loading test data')
train_data_meta, train_data, test_dataset = build_dataset(dataset, 10)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=100, shuffle=False, **kwargs)

print('load test data successfully')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()


    model = build_model()

    net_dict = torch.load(args.checkpoint_path)

    model.load_state_dict(net_dict['state_dict'])

    prec1, preds, gt_labels = validate(
            test_loader, model, nn.CrossEntropyLoss().cuda(), 0)
    print('Test result:\n'
            'Dataset: {0}\t'
            'Imb_factor: {1}\t'
            'Accuracy: {2:.2f} \t'
            'Error: {3:.2f} \n'.format(
                dataset, int(1 / imb_factor), prec1,100 - prec1))



def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    true_labels = []
    preds = []

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        with torch.no_grad():
            _, output = model(input_var)

        output_numpy = output.data.cpu().numpy()
        preds_output = list(output_numpy.argmax(axis=1))

        true_labels += list(target_var.data.cpu().numpy())
        preds += preds_output

        prec1 = accuracy(output.data, target, topk=(1,))[0]
        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, preds, true_labels


def build_model():
    model = ResNet32(dataset == 'cifar10' and 10 or 100)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model



class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




if __name__ == '__main__':
    main()

