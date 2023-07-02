import os
import numpy as np
import argparse
import errno
from collections import OrderedDict
from tqdm import tqdm
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from lib.data.dataset_action import NTURGBD

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

BATCH_SIZE = 128

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/action/MB_train_NTU60_xsub.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    opts = parser.parse_args()
    return opts

def _output_representation(test_loader, backbone_model):
    backbone_model.eval()
    # concat the output of all batches
    output_all = torch.Tensor()
    with torch.no_grad():
        for idx, (batch_input, batch_gt) in tqdm(enumerate(test_loader)):
            # print(batch_input.shape) # [128, 243, 17, 3]
            if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input = batch_input.cuda()
            output = backbone_model.module.get_representation(batch_input)
            output = output.cpu()
            output_all = torch.cat((output_all, output), dim=0)
            print("progress: ", idx, " / ", len(test_loader))
    return output_all # instances, frames(243), joints(17), channels(512)

def output_representation(args, opts):
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    model_backbone = load_backbone(args)

    chk_filename = os.path.join(opts.checkpoint, "latest_epoch_lite.bin")
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_pos']
    model_backbone = load_pretrained_weights(model_backbone, checkpoint)
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    print('Loading dataset...')
    trainloader_params = {
          'batch_size': BATCH_SIZE,
          'shuffle': True,
          'num_workers': 2,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    testloader_params = {
          'batch_size': BATCH_SIZE,
          'shuffle': False,
          'num_workers': 2,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    data_path = 'data/action/%s.pkl' % args.dataset
    ntu60_xsub_train = NTURGBD(data_path=data_path, data_split=args.data_split+'_train', n_frames=args.clip_len, random_move=args.random_move, scale_range=args.scale_range_train)
    ntu60_xsub_val = NTURGBD(data_path=data_path, data_split=args.data_split+'_val', n_frames=args.clip_len, random_move=False, scale_range=args.scale_range_test)

    train_loader = DataLoader(ntu60_xsub_train, **trainloader_params)
    test_loader = DataLoader(ntu60_xsub_val, **testloader_params)
    
    all_output = _output_representation(test_loader, model_backbone)
    print(all_output.shape)

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    output_representation(args, opts)