import argparse
import waveglow.arg_parser

import torch
from torch.utils.data import DataLoader
import numpy as np
from data.dataset import VCDataset
from data.Sample_dataset import MultiSpeaker
from data import collate_fn
from model_vc import Generator
import os
from audioUtils.hparams import hparams
import torch.nn as nn


def load_dataset(data_roots):
    dataloader = DataLoader(VCDataset(data_roots), batch_size=32, shuffle=True, num_workers=32, collate_fn=collate_fn)
    return dataloader


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  Creating %s...  ---" % path)
        print("---  OK  ---")

    else:
        print("---  %s already exists!  ---" % path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train', default=True, action='store_true')
    # parser.add_argument('--test', default=False, action='store_true')
    # parser.add_argument('--load_model', default=False, action='store_true')
    # parser.add_argument('-load_model_path', default='/storage/model/voice_conversion/'
    #                                                 'pretrain_model.pkl-19999')
    parser.add_argument('--data_path', nargs='+', required=True)
    parser.add_argument('--experiment_name', required=True)
    # parser.add_argument('--dis', default=False, type=bool)
    parser.add_argument('--dis', dest='dis', default=False, action='store_true')
    # parser.add_argument('--use_lsgan', default=True, type=bool)
    parser.add_argument('--use_lsgan', dest='use_lsgan', default=False, action='store_true')
    parser.add_argument('--lambda_gan', default=0.01, type=float)
    parser.add_argument('--encoder_type', default='nospeaker')
    parser.add_argument('--decoder_type', default='simple')
    parser.add_argument('--num_speakers', default=2, type=int)
    # parser.add_argument('--loss_content', default=False)
    parser.add_argument('--loss_content', dest='loss_content', default=False, action='store_true')
    # parser.add_argument('--cycle', default=False)
    parser.add_argument('--cycle', dest='cycle', default=False, action='store_true')
    parser.add_argument('--idt_type', default='L1')
    # parser.add_argument('--multigpu', default=True)
    parser.add_argument('--multigpu', dest='multigpu', default=False, action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--vocoder_type', default='griffin')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--save_freq', default=500, type=int)
    parser.add_argument('--display_freq', default=10, type=int)
    parser.add_argument('--lambda_wavenet', default=0.01, type=float)
    parser.add_argument('--dim_spec', default=80, type=int)
    parser.add_argument('--mode', default='mel')
    parser.add_argument('--test_path_source', default=None)
    parser.add_argument('--test_path_target', default=None)
    parser.add_argument('--load_model', default=None)
    parser.add_argument('--initial_iter', default=0, type=int)
    parser.add_argument('--token_num', default=64, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--phoneme_token', dest='phoneme_token', default=False, action='store_true')
    parser.add_argument('--freq', default=32, type=int)
    # parser.add_argument('-dataset_path', default='/storage/raw_feature/voice_conversion/vctk/vctk.h5')
    # parser.add_argument('-index_path', default='/storage/raw_feature/voice_conversion/vctk/128_513_2000k.json')
    # parser.add_argument('-output_model_path', default='/storage/model/voice_conversion/model.pkl')

    parser = waveglow.arg_parser.parse_waveglow_args(parser)

    args = parser.parse_args()

    print(args)

    dataloader = MultiSpeaker(args.data_path, batch_size=args.batch_size, num_workers=8, ret_wav=True)

    if args.multigpu:
        device = 'cuda:0'
    else:
        device = args.device

    experimentName = args.experiment_name
    save_dir = "/mnt/lustre/dengkangle/cmu/saved_models/" + experimentName
    mkdir("logs/" + experimentName)
    mkdir("/mnt/lustre/dengkangle/cmu/saved_models/" + experimentName)
    G = Generator(hparams.dim_neck, hparams.speaker_embedding_size, 512, hparams.freq, lr=1e-3, is_train=True,
                  decoder_type=args.decoder_type,
                  encoder_type=args.encoder_type,
                  num_speakers=args.num_speakers,
                  loss_content=args.loss_content,
                  discriminator=args.dis,
                  use_lsgan=args.use_lsgan,
                  lambda_gan=args.lambda_gan,
                  cycle=args.cycle,
                  idt_type=args.idt_type,
                  multigpu=args.multigpu,
                  vocoder_type=args.vocoder_type,
                  train_wavenet=True,
                  lambda_wavenet=args.lambda_wavenet,
                  test_path_source=args.test_path_source,
                  test_path_target=args.test_path_target,
                  args=args).to(device)

    # G.test_wavenet(dataloader, args.epochs, device, experimentName=experimentName, save_dir=save_dir)
    # G.test_waveglow(dataloader, args.epochs, device, experimentName=experimentName, save_dir=save_dir)
    G.optimize_parameters(dataloader, args.epochs, device, experimentName=experimentName, save_dir=save_dir,
                          save_freq=args.save_freq, display_freq=args.display_freq,
                          load_model=args.load_model,
                          initial_niter=args.initial_iter, withmask=False, test_fixed=True)



