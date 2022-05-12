CUDA_VISIBLE_DEVICES=7 python3 test.py --loading_path checkpoints/MetaSAug_ImageNet_LT.tar --dataset ImageNet_LT --num_classes 1000 --data_root ../ImageNet
CUDA_VISIBLE_DEVICES=7 python3 test.py --loading_path checkpoints/MetaSAug_iNat18.tar --dataset iNaturalist18 --num_classes 8142 --data_root ../iNaturalist18
