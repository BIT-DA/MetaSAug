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
from resnet import *
import shutil
from loss import *

parser = argparse.ArgumentParser(description='Imbalanced Example')
parser.add_argument('--dataset', default='cifar100', type=str,
                    help='dataset (cifar10 or cifar100[default])')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--num_meta', type=int, default=10,
                    help='The number of meta data for each class.')
parser.add_argument('--imb_factor', type=float, default=0.005)
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--split', type=int, default=1000)
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--lam', default=0.25, type=float, help='[0.25, 0.5, 0.75, 1.0]')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--meta_lr', default=0.1, type=float)
parser.add_argument('--save_name', default='name', type=str)
parser.add_argument('--idx', default='0', type=str)


args = parser.parse_args()
for arg in vars(args):
    print("{}={}".format(arg, getattr(args, arg)))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)
kwargs = {'num_workers': 1, 'pin_memory': False}
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

train_data_meta, train_data, test_dataset = build_dataset(args.dataset, args.num_meta)

print(f'length of meta dataset:{len(train_data_meta)}')
print(f'length of train dataset: {len(train_data)}')

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

np.random.seed(42)
random.seed(42)
torch.manual_seed(args.seed)
classe_labels = range(args.num_classes)

data_list = {}


for j in range(args.num_classes):
    data_list[j] = [i for i, label in enumerate(train_loader.dataset.targets) if label == j]


img_num_list = get_img_num_per_cls(args.dataset, args.imb_factor, args.num_meta*args.num_classes)
print(img_num_list)
print(sum(img_num_list))

im_data = {}
idx_to_del = []
for cls_idx, img_id_list in data_list.items():
    random.shuffle(img_id_list)
    img_num = img_num_list[int(cls_idx)]
    im_data[cls_idx] = img_id_list[img_num:]
    idx_to_del.extend(img_id_list[img_num:])

print(len(idx_to_del))
imbalanced_train_dataset = copy.deepcopy(train_data)
imbalanced_train_dataset.targets = np.delete(train_loader.dataset.targets, idx_to_del, axis=0)
imbalanced_train_dataset.data = np.delete(train_loader.dataset.data, idx_to_del, axis=0)
print(len(imbalanced_train_dataset))
imbalanced_train_loader = torch.utils.data.DataLoader(
    imbalanced_train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

validation_loader = torch.utils.data.DataLoader(
    train_data_meta, batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

best_prec1 = 0

beta = 0.9999
effective_num = 1.0 - np.power(beta, img_num_list)
per_cls_weights = (1.0 - beta) / np.array(effective_num)
per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(img_num_list)
per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
weights = torch.tensor(per_cls_weights).float()

def main():
    global args, best_prec1
    args = parser.parse_args()

    model = build_model()
    optimizer_a = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)

    cudnn.benchmark = True

    criterion = LDAM_meta(64, args.dataset == "cifar10" and 10 or 100, cls_num_list=img_num_list,
                                    max_m=0.5, s=30)

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer_a, epoch + 1)

        ratio = args.lam * float(epoch) / float(args.epochs)
        if epoch < 160:
            train(imbalanced_train_loader, model, optimizer_a, epoch)

        else:
            train_MetaSAug(imbalanced_train_loader, validation_loader, model, optimizer_a, epoch, criterion, ratio)

       
        prec1, preds, gt_labels = validate(test_loader, model, nn.CrossEntropyLoss().cuda(), epoch)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        
        # save_checkpoint(args, {
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'best_acc1': best_prec1,
        #     'optimizer': optimizer_a.state_dict(),
        # }, is_best)

    print('Best accuracy: ', best_prec1)


def train(train_loader, model, optimizer_a, epoch):

    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    for i, (input, target) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        _, y_f = model(input_var)
        del _
        cost_w = F.cross_entropy(y_f, target_var, reduce=False)
        l_f = torch.mean(cost_w)
        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]


        losses.update(l_f.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))

        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses,top1=top1))



def train_MetaSAug(train_loader, validation_loader, model,optimizer_a, epoch, criterion, ratio):

    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    for i, (input, target) in enumerate(train_loader):

        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        cv = criterion.get_cv()
        cv_var = to_var(cv)

        meta_model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
        meta_model.load_state_dict(model.state_dict())
        meta_model.cuda()

        feat_hat, y_f_hat = meta_model(input_var)
        cls_loss_meta = criterion(meta_model.linear, feat_hat, y_f_hat, target_var, ratio,
                                  weights, cv_var, "none")
        meta_model.zero_grad()

        grads = torch.autograd.grad(cls_loss_meta, (meta_model.params()), create_graph=True)
        meta_lr = args.lr * ((0.01 ** int(epoch >= 160)) * (0.01 ** int(epoch >= 180)))
        meta_model.update_params(meta_lr, source_params=grads)

        input_val, target_val = next(iter(validation_loader))
        input_val_var = to_var(input_val, requires_grad=False)
        target_val_var = to_var(target_val, requires_grad=False)

        _, y_val = meta_model(input_val_var)
        cls_meta = F.cross_entropy(y_val, target_val_var)
        grad_cv = torch.autograd.grad(cls_meta, cv_var, only_inputs=True)[0]
        new_cv = cv_var - args.meta_lr * grad_cv

        del grad_cv, grads

        #model.train()
        features, predicts = model(input_var)
        cls_loss = criterion(model.linear, features, predicts, target_var, ratio, weights, new_cv, "update")

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

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, preds, true_labels


def build_model():
    model = ResNet32(args.dataset == 'cifar10' and 10 or 100)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True


    return model

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


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


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * ((0.01 ** int(epoch >= 160)) * (0.01 ** int(epoch >= 180)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


def save_checkpoint(args, state, is_best):
    path = 'checkpoint/ours/' + args.idx + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + args.save_name + '_ckpt.pth.tar'
    if is_best:
        torch.save(state, filename)

if __name__ == '__main__':
    main()
