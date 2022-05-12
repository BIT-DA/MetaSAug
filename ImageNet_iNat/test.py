
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
import torchvision.transforms as transforms
from data_utils import *
from dataloader import load_data_distributed
import shutil
from ResNet import *
import resnet_meta

import multiprocessing
import torch.nn.parallel
import torch.nn as nn
from collections import Counter
import time
parser = argparse.ArgumentParser(description='Imbalanced Example')
parser.add_argument('--dataset', default='iNaturalist18', type=str,
                    help='dataset')
parser.add_argument('--data_root', default='/data1/TL/data/iNaturalist18', type=str)
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_classes', type=int, default=8142)
parser.add_argument('--num_meta', type=int, default=10,
                    help='The number of meta data for each class.')
parser.add_argument('--test_batch_size', type=int, default=512, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--workers', default=16, type=int)
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--split', type=int, default=1000)
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--print-freq', '-p', default=1000, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--gpu', default=None, type=int)
parser.add_argument('--lam', default=0.25, type=float)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--meta_lr', default=0.1, type=float)
parser.add_argument('--loading_path', default=None, type=str)

args = parser.parse_args()
# print(args)
for arg in vars(args):
    print("{}={}".format(arg, getattr(args, arg)))


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

kwargs = {'num_workers': 1, 'pin_memory': True}
use_cuda = not args.no_cuda and torch.cuda.is_available()

cudnn.benchmark = True
cudnn.enabled = True
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

if args.dataset == 'ImageNet_LT':
    val_set = load_data_distributed(data_root=args.data_root, dataset=args.dataset, phase="test", batch_size=args.test_batch_size,
                    num_workers=args.workers, test_open=False, shuffle=False)
else:
    val_set = load_data_distributed(data_root=args.data_root, dataset=args.dataset, phase="val", batch_size=args.test_batch_size,
                    num_workers=args.workers, test_open=False, shuffle=False)

val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)


np.random.seed(42)
random.seed(42)
torch.manual_seed(args.seed)

data_list = {}
data_list_num = []
best_prec1 = 0

model_dict = {"ImageNet_LT": "models/resnet50_uniform_e90.pth",
            "iNaturalist18": "models/iNat18/resnet50_uniform_e200.pth"}

def main():
    global args, best_prec1
    args = parser.parse_args()
    
    cudnn.benchmark = True

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    model = FCModel(2048, args.num_classes)
    model = model.cuda()
    loading_path = args.loading_path
    weights = torch.load(loading_path, map_location=torch.device("cpu"))   
    weights_c = weights['state_dict']['classifier']
    weights_c = {k: weights_c[k] for k in model.state_dict()}
    for k in model.state_dict():
        if k not in weights:
            print("Loading Weights Warning.")

    model.load_state_dict(weights_c) 
    feature_extractor = create_model(stage1_weights=True,  dataset=args.dataset, log_dir=model_dict[args.dataset])
    weight_f = weights['state_dict']['feature']
    feature_extractor.load_state_dict(weight_f)
    feature_extractor = feature_extractor.cuda()
    feature_extractor.eval()

    prec1, preds, gt_labels = validate(val_loader, model, feature_extractor,  nn.CrossEntropyLoss())

    print('Accuracy: ', prec1)

def validate(val_loader, model, feature_extractor, criterion):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    true_labels = []
    preds = []

    torch.cuda.empty_cache()
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # compute output
        with torch.no_grad():
            feature = feature_extractor(input)
            output = model(feature)
        loss = criterion(output, target)

        output_numpy = output.data.cpu().numpy()
        preds_output = list(output_numpy.argmax(axis=1))

        true_labels += list(target.data.cpu().numpy())
        preds += preds_output


        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    # import pdb; pdb.set_trace()

    return top1.avg, preds, true_labels



def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

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
    """Computes the precision@k for the specified values of k"""
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


def save_checkpoint(args, state, is_best, epoch):
    filename =  'checkpoint/' + 'train_' + str(args.dataset) + '/' + str(args.lr) + '_' + str(args.batch_size) + '_' + str(args.meta_lr) + 'epoch' + str(epoch) + '_ckpt.pth.tar'
    file_root, _ = os.path.split(filename)
    if not os.path.exists(file_root):
        os.makedirs(file_root)
    torch.save(state, filename)


if __name__ == '__main__':
    main()
