# MetaSAug Implemented in PyTorch

## Prerequisite

- PyTorch >= 1.2.0
- Python3
- torchvision
- PIL
- argparse
- numpy

## Evaluation

For faster evaluation, we provide several pre-trained models of MetaSAug. We can use `MetaSAug_test_CE.sh` & `MetaSAug_test_LDAM.sh` to test MetaSAug with cross-entropy loss and LDAM loss, respectively. The models are stored in `checkpoints/ours`.

Evaluation examples: 

- `sh MetaSAug_test_CE.sh`
- `sh MetaSAug_test_LDAM.sh`

## Training example

```
CIFAR-LT-100, MetaSAug with LDAM loss
python3.6 MetaSAug_LDAM_train.py --gpu 0 --lr 0.1 --lam 0.75 --imb_factor 0.05 --dataset cifar100 --num_classes 100 --save_name MetaSAug_cifar100_LDAM_imb0.05 --idx 1
```



