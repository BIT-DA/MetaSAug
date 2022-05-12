# MetaSAug: Meta Semantic Augmentation for Long-Tailed Visual Recognition

This repository contains the code for:

**MetaSAug: Meta Semantic Augmentation for Long-Tailed Visual Recognition**.

Shuang Li, Kaixiong Gong, et al.

IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2021. [[arXiv](https://arxiv.org/abs/2103.12579)]  [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_MetaSAug_Meta_Semantic_Augmentation_for_Long-Tailed_Visual_Recognition_CVPR_2021_paper.pdf)]

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

- `sh scripts/MetaSAug_CE_test.sh`
- `sh scripts/MetaSAug_LDAM_test.sh`

## Training example

```
CIFAR-LT-100, MetaSAug with LDAM loss
python3.6 MetaSAug_LDAM_train.py --gpu 0 --lr 0.1 --lam 0.75 --imb_factor 0.05 --dataset cifar100 --num_classes 100 --save_name MetaSAug_cifar100_LDAM_imb0.05 --idx 1
```

Or run:

```
sh scripts/MetaSAug_LDAM_train.sh
```

## Results and models
**CIFAR-LT-10**
|   Model   | Imbalance| Top-1 Error |   Download    |Model   | Imbalance| Top-1 Error |   Download    |
| --------- |:--------:|:-----------:|:-------------:|--------- |:--------:|:-----------:|:-------------:|
| MetaSAug+LDAM | 200  |22.65 | [ResNet32](https://github.com/BIT-DA/MetaSAug/tree/main/checkpoint/ours) |MetaSAug+CE | 200  |23.11 | [ResNet32](https://github.com/BIT-DA/MetaSAug/tree/main/checkpoint/ours) |
| MetaSAug+LDAM | 100  |19.34 | [ResNet32](https://github.com/BIT-DA/MetaSAug/tree/main/checkpoint/ours) |MetaSAug+CE | 200  |19.46 | [ResNet32](https://github.com/BIT-DA/MetaSAug/tree/main/checkpoint/ours) |
| MetaSAug+LDAM | 50  |15.66 | [ResNet32](https://github.com/BIT-DA/MetaSAug/tree/main/checkpoint/ours) |MetaSAug+CE | 200  |15.97 | [ResNet32](https://github.com/BIT-DA/MetaSAug/tree/main/checkpoint/ours) |
| MetaSAug+LDAM | 20  |11.90  | [ResNet32](https://github.com/BIT-DA/MetaSAug/tree/main/checkpoint/ours) |MetaSAug+CE | 200  |12.36 | [ResNet32](https://github.com/BIT-DA/MetaSAug/tree/main/checkpoint/ours) |
| MetaSAug+LDAM | 10  |10.32 | [ResNet32](https://github.com/BIT-DA/MetaSAug/tree/main/checkpoint/ours) |MetaSAug+CE | 200  |10.56 | [ResNet32](https://github.com/BIT-DA/MetaSAug/tree/main/checkpoint/ours) |

**CIFAR-LT-100**
|   Model   | Imbalance| Top-1 Error |   Download    |Model   | Imbalance| Top-1 Error |   Download    |
| --------- |:--------:|:-----------:|:-------------:|--------- |:--------:|:-----------:|:-------------:|
| MetaSAug+LDAM | 200  |56.91 | [ResNet32](https://github.com/BIT-DA/MetaSAug/tree/main/checkpoint/ours) |MetaSAug+CE | 200  |60.06 | [ResNet32](https://github.com/BIT-DA/MetaSAug/tree/main/checkpoint/ours) |
| MetaSAug+LDAM | 100  |51.99 | [ResNet32](https://github.com/BIT-DA/MetaSAug/tree/main/checkpoint/ours) |MetaSAug+CE | 200  |53.13 | [ResNet32](https://github.com/BIT-DA/MetaSAug/tree/main/checkpoint/ours) |
| MetaSAug+LDAM | 50  |47.73 | [ResNet32](https://github.com/BIT-DA/MetaSAug/tree/main/checkpoint/ours) |MetaSAug+CE | 200  |48.10 | [ResNet32](https://github.com/BIT-DA/MetaSAug/tree/main/checkpoint/ours) |
| MetaSAug+LDAM | 20  |42.47  | [ResNet32](https://github.com/BIT-DA/MetaSAug/tree/main/checkpoint/ours) |MetaSAug+CE | 200  |42.15 | [ResNet32](https://github.com/BIT-DA/MetaSAug/tree/main/checkpoint/ours) |
| MetaSAug+LDAM | 10  |38.72 | [ResNet32](https://github.com/BIT-DA/MetaSAug/tree/main/checkpoint/ours) |MetaSAug+CE | 200  |38.27 | [ResNet32](https://github.com/BIT-DA/MetaSAug/tree/main/checkpoint/ours) |

## Acknowledgements
Some codes in this project are adapted from [Meta-class-weight](https://github.com/abdullahjamal/Longtail_DA). We thank them for their excellent projects.

## Citation
If you find this code useful for your research, please cite our paper:
```
@inproceedings{li2021MetaSAug,
author = {Li, Shuang and Gong, Kaixiong and Liu, Chi Harold and Wang, Yulin and Qiao, Feng and Cheng, Xinjing},
title = {MetaSAug: Meta Semantic Augmentation for Long-Tailed Visual Recognition},
year = {2021},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
}
```


