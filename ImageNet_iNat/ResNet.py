from resnet_meta import *
from utils import *
from os import path


def create_model(use_selfatt=False, use_fc=False, dropout=None, stage1_weights=False, dataset=None, log_dir=None, test=False, *args):
    
    print('Loading Scratch ResNet 50 Feature Model.')
    if not use_fc:    
        resnet50 = FeatureMeta(BottleneckMeta, [3, 4, 6, 3], dropout=None)
    else:
        resnet50 = FCMeta(2048, 1000)
    if not test:
        if stage1_weights:
            assert dataset
            print('Loading %s Stage 1 ResNet 10 Weights.' % dataset)
            if log_dir is not None:
                # subdir = log_dir.strip('/').split('/')[-1]
                # subdir = subdir.replace('stage2', 'stage1')
                # weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), subdir)
                #weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), 'stage1')
                weight_dir = log_dir
            else:
                weight_dir = './logs/%s/stage1' % dataset
            print('==> Loading weights from %s' % weight_dir)
            if not use_fc:
                resnet50 = init_weights(model=resnet50,
                                        weights_path=weight_dir)
            else:
                resnet50 = init_weights(model=resnet50, weights_path=weight_dir, classifier=True)
            #resnet50.load_state_dict(torch.load(weight_dir))
        else:
            print('No Pretrained Weights For Feature Model.')

    return resnet50
