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
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import h5py


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
    output_all = []
    label_all = []
    with torch.no_grad():
        for idx, (batch_input, batch_gt) in tqdm(enumerate(test_loader)):
            # print(batch_input.shape) # [128, 243, 17, 3]
            if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input = batch_input.cuda()
            output = backbone_model.module.get_representation(batch_input)
            output = output.cpu().detach()

            output_all.append(output)
            # output_all = torch.cat((output_all, output), dim=0)
            # and ground truth gt
            batch_gt = batch_gt.cpu()
            label_all.append(batch_gt)
            print("progress: ", idx, " / ", len(test_loader))
        output_all = torch.cat(output_all, dim=0)
        label_all = torch.cat(label_all, dim=0)
    return output_all, label_all # instances, frames(243), joints(17), channels(512)

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
          'shuffle': False,
          'num_workers': 1,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    testloader_params = {
          'batch_size': BATCH_SIZE,
          'shuffle': False,
          'num_workers': 1,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    data_path = 'data/action/%s.pkl' % args.dataset
    ntu60_xsub_train = NTURGBD(data_path=data_path, data_split=args.data_split+'_val', n_frames=args.clip_len, random_move=False, scale_range=args.scale_range_test)
    ntu60_xsub_val = NTURGBD(data_path=data_path, data_split=args.data_split+'_val', n_frames=args.clip_len, random_move=False, scale_range=args.scale_range_test)

    train_loader = DataLoader(ntu60_xsub_train, **trainloader_params)
    test_loader = DataLoader(ntu60_xsub_val, **testloader_params)
    
    # all_output, all_label = _output_representation(train_loader, model_backbone)
    all_output, all_label = _output_representation(test_loader, model_backbone)

    embeddings = np.array([x.numpy() for x in all_output], dtype=object)
    labels = np.array([x.numpy() for x in all_label], dtype=object)
    # np.save(os.path.join(opts.checkpoint, "embeddings.npy"), embeddings)
    # np.save(os.path.join(opts.checkpoint, "labels.npy"), labels)
    batch_size = 1000
    num_batches = len(embeddings) // batch_size

    with h5py.File(os.path.join(opts.checkpoint, "embeddings.h5"), "w") as f:
        f.create_dataset("embeddings", data=embeddings)

def visualize_representation(features, labels):
    # Reshape the features to a 2D array of shape (N*243, 17*512)
    flat_features = features.reshape((features.shape[0], -1))


    # # Standardize the features
    # mean = np.mean(flat_features, axis=0)
    # std = np.std(flat_features, axis=0)
    # std_features = (flat_features - mean) / std

    # Apply t-SNE to reduce the dimensionality to 2 dimensions
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=30)
    embedded_data = tsne.fit_transform(flat_features)

    # Plot the reduced features using a scatter plot
    plt.figure(figsize=(10, 5))
    plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=labels)
    plt.colorbar()
    plt.savefig("tsne.png")
    print("Saved tsne.png")
    
if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)

    output_representation(args, opts)

    # # load embeddings.npy and labels.npy
    # print("Loading embeddings.npy and labels.npy...")
    # features = []
    # labels = []
    # for i in range(10):
    #     features.append(np.load(os.path.join(opts.checkpoint, f"embeddings_{i}.npy"), allow_pickle=True))
    #     labels.append(np.load(os.path.join(opts.checkpoint, f"labels_{i}.npy"), allow_pickle=True))
    # features = np.concatenate(features, axis=0)
    # labels = np.concatenate(labels, axis=0)
    # print("features.shape: ", features.shape)
    # print("labels.shape: ", labels.shape)
    
    # features = np.load(os.path.join(opts.checkpoint, "embeddings.npy"), allow_pickle=True)
    # labels = np.load(os.path.join(opts.checkpoint, "labels.npy"), allow_pickle=True)
    # print("features.shape: ", features.shape)
    # print("labels.shape: ", labels.shape)
    # print("Visualizing...")
    # visualize_representation(features, labels)