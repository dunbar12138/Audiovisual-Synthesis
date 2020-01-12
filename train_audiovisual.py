import argparse

import torch
from torch.utils.data import DataLoader
import numpy as np
from data.Sample_dataset import SampleVideoDataset
from data import collate_fn
from model_vc import VideoAudioGenerator
import os
from audioUtils.hparams import hparams
import torch.nn as nn


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  Creating %s...  ---" % path)
        print("---  OK  ---")

    else:
        print("---  %s already exists!  ---" % path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', required=True)
       
    parser.add_argument('--multigpu', dest='multigpu', default=False, action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--save_freq', default=500, type=int)
    parser.add_argument('--display_freq', default=10, type=int)
    parser.add_argument('--lambda_wavenet', default=0.01, type=float)
    parser.add_argument('--load_model', default=None)
    parser.add_argument('--initial_iter', default=0, type=int)
    parser.add_argument('--attention_map', default=None)
    parser.add_argument('--residual', dest='residual', default=False, action='store_true')
    parser.add_argument('--video_path', default=None)
    parser.add_argument('--use_256', dest='use_256', default=False, action='store_true')
    parser.add_argument('--loss_content', dest='loss_content', default=False, action='store_true')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--test_path', default=None)

    args = parser.parse_args()

    print(args)

    if args.video_path is None:
        dataloader = DataLoader(SampleVideoDataset("/mnt/lustre/dengkangle/cmu/datasets/video/obama_train.mp4", ret_wav=True, use_256=args.use_256), batch_size=args.batch_size, shuffle=True, num_workers=8)
    else:
        dataloader = DataLoader(SampleVideoDataset(args.video_path, ret_wav=True, use_256=args.use_256), batch_size=args.batch_size, shuffle=True, num_workers=8)

    if args.multigpu:
        device = 'cuda:0'
    else:
        device = args.device

    experimentName = args.experiment_name
    save_dir = os.path.join(args.save_dir, experimentName)
    mkdir("logs/" + experimentName)
    mkdir(save_dir)
    G = VideoAudioGenerator(hparams.dim_neck, hparams.speaker_embedding_size, 512, hparams.freq, lr=1e-3, is_train=True,
                  multigpu=args.multigpu,
                  lambda_wavenet=args.lambda_wavenet,
                  residual=args.residual,
                  attention_map=args.attention_map,
                  use_256=args.use_256,
                  loss_content=args.loss_content,
                  test_path = args.test_path).to(device)

    G.optimize_parameters_video(dataloader, args.epochs, device, display_freq=args.display_freq, save_freq=args.save_freq, save_dir=save_dir, experimentName=experimentName, initial_niter=args.initial_iter, load_model=args.load_model)

