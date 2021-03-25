# MetaSAug: Meta Semantic Augmentation for Long-Tailed Visual Recognition

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

## Acknowledgements
Some codes in this project are adapted from [Meta-class-weight](https://github.com/abdullahjamal/Longtail_DA). We thank them for their excellent projects.

## Citation
If you find this code useful for your research, please cite our paper:
```
@inproceedings{li2019MetaSAug,
author = {Shuang Li and Kaixiong Gong and Chi Harold Liu and Yulin Wang and Feng Qiao and Xinjing Chen},
title = {MetaSAug: Meta Semantic Augmentation for Long-Tailed Visual Recognition},
year = {2021},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
}
```


