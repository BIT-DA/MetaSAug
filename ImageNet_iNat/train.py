
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
# import resnet
from dataloader import load_data_distributed
import shutil
from ResNet import *
import loss
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
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

print(f'num_gpus: {num_gpus}')
args.distributed = num_gpus > 1
print("ditributed: {args.distributed}")
if args.distributed:
    torch.cuda.set_device(args.local_rank)
    #torch.distributed.init_process_group(backend="nccl", init_method="env://")
    torch.distributed.init_process_group(backend="nccl")
    args.batch_size = int(args.batch_size / num_gpus)


######### ImageNet dataset
splits = ["train", "val", "test"]
if args.dataset == 'ImageNet_LT':
    train_set = load_data_distributed(data_root=args.data_root, dataset=args.dataset, phase="train", batch_size=args.batch_size,
                    num_workers=args.workers, test_open=False, shuffle=False)
    val_set = load_data_distributed(data_root=args.data_root, dataset=args.dataset, phase="test", batch_size=args.test_batch_size,
                    num_workers=args.workers, test_open=False, shuffle=False)

    meta_set =  load_data_distributed(data_root=args.data_root, dataset=args.dataset, phase="val", batch_size=args.batch_size, num_workers=args.workers, test_open=False, shuffle=False)

else:
    train_set = load_data_distributed(data_root=args.data_root, dataset=args.dataset, phase="train", batch_size=args.batch_size,
                    num_workers=args.workers, test_open=False, shuffle=False)
    val_set = load_data_distributed(data_root=args.data_root, dataset=args.dataset, phase="val", batch_size=args.test_batch_size,
                    num_workers=args.workers, test_open=False, shuffle=False)
    meta_set = train_set

if args.dataset == 'iNaturalist17':
    meta_set, _ = build_dataset(meta_set, 5, args.num_classes) 
elif args.dataset == 'iNaturalist18':
    meta_set, _ = build_dataset(meta_set, 2, args.num_classes)
else:
    meta_set, _ = build_dataset(meta_set, 10, args.num_classes)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=0, 
        pin_memory=True, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
meta_sampler = torch.utils.data.distributed.DistributedSampler(meta_set)
meta_loader = torch.utils.data.DataLoader(meta_set, batch_size=args.batch_size, shuffle=(meta_sampler is None), num_workers=0,
        pin_memory=True, sampler=meta_sampler)


np.random.seed(42)
random.seed(42)
torch.manual_seed(args.seed)
classe_labels = range(args.num_classes)

data_list = {}
data_list_num = []
num = Counter(train_loader.dataset.labels)
data_list_num = [0] * args.num_classes
for key in num:
    data_list_num[key] = num[key]

beta = 0.9999
effective_num = 1.0 - np.power(beta, data_list_num)
per_cls_weights = (1.0 - beta) / np.array(effective_num)
per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(data_list_num)
per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

model_dict = {"ImageNet_LT": "models/resnet50_uniform_e90.pth",
            "iNaturalist18": "models/iNat18/resnet50_uniform_e200.pth"}


def main():
    global args
    args = parser.parse_args()
    
    cudnn.benchmark = True
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(f'local_rank: {args.local_rank}')
    model = FCModel(2048, args.num_classes)
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    weights = torch.load(model_dict[args.dataset], map_location=torch.device("cpu"))   
    weights = weights['state_dict_best']['classifier']
    weights = {k: weights['module.' + k] for k in model.module.state_dict()}
    for k in model.module.state_dict():
        if k not in weights:
            print("Pretrained Weights Warning.")

    model.module.load_state_dict(weights) 
    feature_extractor = create_model(stage1_weights=True,  dataset=args.dataset, log_dir=model_dict[args.dataset])
    feature_extractor = feature_extractor.cuda()
    feature_extractor.eval()


    torch.autograd.set_detect_anomaly(True)
    torch.distributed.barrier()

    optimizer_a = torch.optim.SGD(model.module.parameters(), args.lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)

    criterion = loss.Loss_meta(2048, args.num_classes)
    for epoch in range(args.epochs):
        ratio = args.lam * float(epoch) / float(args.epochs)
        train_meta(train_loader, model, feature_extractor,  optimizer_a, epoch, criterion, ratio)
        
        if args.local_rank == 0:
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'state_dict': {'feature': feature_extractor.state_dict(), 'classifier': model.module.state_dict()},
                'optimizer' : optimizer_a.state_dict(),
            }, False, epoch)

def train_meta(train_loader, model, feature_extractor, optimizer_a, epoch, criterion, ratio):
    """Experimenting how to train stably in stage-2"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    meta_losses = AverageMeter()
    top1 = AverageMeter()
    meta_top1 = AverageMeter()
    model.train()
    weights = torch.tensor(per_cls_weights).float()
    for i, (input, target) in enumerate(train_loader):

        input_var = input.cuda(non_blocking=True)
        target_var = target.cuda(non_blocking=True)
        cv = criterion.get_cv()
        cv_var = to_var(cv)

        meta_model = FCMeta(2048, args.num_classes)
        meta_model.load_state_dict(model.module.state_dict())
        meta_model.cuda()

        with torch.no_grad():
            feat_hat = feature_extractor(input_var)
        y_f_hat = meta_model(feat_hat) 
        cls_loss_meta = criterion(list(meta_model.fc.named_leaves())[0][1], feat_hat, y_f_hat, target_var, ratio,
                                  weights, cv_var, "none")
        meta_model.zero_grad()
        grads = torch.autograd.grad(cls_loss_meta, (meta_model.params()), create_graph=True)
        meta_lr = args.lr 
        meta_model.fc.update_params(meta_lr, source_params=grads)

        input_val, target_val = next(iter(meta_loader))
        input_val_var = input_val.cuda(non_blocking=True)
        target_val_var = target_val.cuda(non_blocking=True)
        
        with torch.no_grad():
            feature_val = feature_extractor(input_val_var)
        y_val = meta_model(feature_val)
        cls_meta = F.cross_entropy(y_val, target_val_var)
        grad_cv = torch.autograd.grad(cls_meta, cv_var, only_inputs=True)[0]
        new_cv = cv - args.meta_lr * grad_cv  

        del grad_cv, grads, meta_model
        with torch.no_grad():
            features = feature_extractor(input_var)
        predicts = model(features)
        cls_loss = criterion(list(model.module.fc.parameters())[0], features, predicts, target_var, ratio, weights, new_cv.detach(), "update")

        prec_train = accuracy(predicts.data, target_var.data, topk=(1,))[0]

        losses.update(cls_loss.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))

        optimizer_a.zero_grad()
        cls_loss.backward()
        optimizer_a.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses,top1=top1))



def validate(val_loader, model, feature_extractor, criterion, epoch, local_rank, distributed):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode

    true_labels = []
    preds = []
    if distributed:
        model = model.module
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

        if i % args.print_freq == 0 and local_rank==0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
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
