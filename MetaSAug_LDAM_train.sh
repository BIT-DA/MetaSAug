CUDA_VISIBLE_DEVICES=0 python3.6 MetaSAug_LDAM_train.py --gpu 0 --lr 0.1 --lam 0.75 --imb_factor 0.05 --dataset cifar100 --num_classes 100 --save_name MetaSAug_cifar100_LDAM_imb0.05 --idx 1
CUDA_VISIBLE_DEVICES=0 python3.6 MetaSAug_LDAM_train.py --gpu 0 --lr 0.1 --lam 0.75 --imb_factor 0.02 --dataset cifar100 --num_classes 100 --save_name MetaSAug_cifar100_LDAM_imb0.02 --idx 1
