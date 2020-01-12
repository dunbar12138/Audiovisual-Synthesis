import argparse

import torch
import numpy as np
from data import collate_fn
from model_vc import VideoAudioGenerator
import os
from audioUtils.hparams import hparams
import torch.nn as nn

from model_video import VideoGenerator
import librosa
from data.Sample_dataset import pad_seq
import imageio

from audioUtils.audio import wav2seg, inv_preemphasis, preemphasis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_path')
    parser.add_argument('--load_model', default=None, required=True)
    parser.add_argument('--output_file')
    parser.add_argument('--residual', dest='residual', default=False, action='store_true')
    parser.add_argument('--multigpu', dest='multigpu', default=False, action='store_true')
    parser.add_argument('--use_256', dest='use_256', default=False, action='store_true')
    args = parser.parse_args()
    print(args)

    device = 'cuda:0'

    load_model = args.load_model
    
    G = VideoAudioGenerator(hparams.dim_neck, hparams.speaker_embedding_size, 512, hparams.freq, lr=1e-3, is_train=False, 
                  multigpu=args.multigpu,
                  residual=args.residual,
                  use_256=args.use_256).to(device)
    
    print("Loading from %s..." % load_model)
    # self.load_state_dict(torch.load(load_model))
    d = torch.load(load_model)
    newdict = d.copy()
    for key, value in d.items():
        newkey = key
        if 'wavenet' in key:
            newdict[key.replace('wavenet', 'vocoder')] = newdict.pop(key)
            newkey = key.replace('wavenet', 'vocoder')
        if not args.multigpu and 'module' in key:
            newdict[newkey.replace('module.','',1)] = newdict.pop(newkey)
            newkey = newkey.replace('module.','', 1)
        if newkey not in G.state_dict():
            newdict.pop(newkey)
    G.load_state_dict(newdict)
    print("AutoVC Model Loaded")

    wav, sr = librosa.load(args.wav_path, hparams.sample_rate)

    mel_basis = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels)
    wav = preemphasis(wav, hparams.preemphasis, hparams.preemphasize)
    linear_spec = np.abs(
        librosa.stft(wav, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size))
    mel_spec = mel_basis.dot(linear_spec)
    mel_db = 20 * np.log10(mel_spec)
    # print(in_fpath, mel_db.min(), mel_db.max())
    test_data = np.clip((mel_db + 120) / 125, 0, 1)
    test_data = torch.Tensor(pad_seq(test_data.T, hparams.freq)).unsqueeze(0).to(device)

    speaker = torch.from_numpy(np.array([0, 1])).float()

    with torch.no_grad():
        mel_outputs_postnet, v_stage1, v_stage2 = G.generate(test_data, speaker, device)

    print(v_stage2.shape)
    gif_list = list(v_stage2.squeeze().cpu().numpy().transpose(0,2,3,1))
    gif_name = "tmp/tmp.gif"

    imageio.mimsave(gif_name, gif_list, 'GIF', duration = 1/20)

    if args.multigpu:
        # s2t_wav = inv_preemphasis(G.vocoder.module.generate(mel_outputs_postnet.transpose(1, 2), True, 8000, 800, mu_law=True), hparams.preemphasis, hparams.preemphasize)
        s2t_wav = inv_preemphasis(G.vocoder.module.generate(mel_outputs_postnet.transpose(1, 2), False, None, None, mu_law=True), hparams.preemphasis, hparams.preemphasize)
    else:
        # s2t_wav = inv_preemphasis(G.vocoder.generate(mel_outputs_postnet.transpose(1, 2), True, 8000, 800, mu_law=True), hparams.preemphasis, hparams.preemphasize)
        s2t_wav = inv_preemphasis(G.vocoder.generate(mel_outputs_postnet.transpose(1, 2), False, None, None, mu_law=True), hparams.preemphasis, hparams.preemphasize)
    
    librosa.output.write_wav("tmp/tmp.wav", s2t_wav.astype(np.float32), hparams.sample_rate)
    os.system(f"ffmpeg -i  tmp/tmp.gif -pix_fmt yuv420p tmp/tmp.mp4 -y; ffmpeg -i tmp/tmp.mp4 -i tmp/tmp.wav -c:v copy -c:a aac -strict experimental {args.output_file} -y")