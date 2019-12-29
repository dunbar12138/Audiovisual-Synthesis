# -*- coding: UTF-8 -*-
import random
import librosa
import torch.utils.data as data
import pickle
import os.path
import glob
import torch
import itertools
from math import ceil
import numpy as np
from scipy.fftpack import dct
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from audioUtils.audio import wav2seg, preemphasis
from vocoder import audio
from audioUtils.hparams import hparams
from data import collate_fn

import imageio
from PIL import Image

from tqdm import tqdm

def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0]) / base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0, len_pad), (0, 0)), 'constant')

class SampleDataset(data.Dataset):
    def __init__(self, npy_path=None, wav_path=None, speaker_id=0, speaker_nums=2, sample_frames=128, length=-1):
        super(SampleDataset, self).__init__()
        if npy_path is not None:
            self.raw_data = np.load(npy_path)
            print('Loading ', npy_path, "\tshape:", self.raw_data.shape)

        elif wav_path is not None:
            print('Encoding ', wav_path)
            wav, sr = librosa.load(wav_path, hparams.sample_rate)
            wav = preemphasis(wav, hparams.preemphasis, hparams.preemphasize)
            wav = wav / (np.abs(wav).max() * 1.1)
            self.wav = audio.encode_mu_law(wav, mu=2 ** hparams.bits)
    
            mel_basis = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels)
            linear_spec = np.abs(
                librosa.stft(wav, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size))
            mel_spec = mel_basis.dot(linear_spec)
            mel_db = 20 * np.log10(mel_spec)
            # print(in_fpath, mel_db.min(), mel_db.max())
            self.raw_data = np.clip((mel_db + 120) / 125, 0, 1)
            print('Raw_Data Shape:', self.raw_data.shape)
            # (num_mels, num_frames)
        else:
            print("Error! No data input...")
        self.speaker = np.zeros(speaker_nums)
        self.speaker[speaker_id % speaker_nums] = 1
        self.sample_frames = sample_frames
        if length > 0:
            self.length = length
        else:
            self.length = max(self.raw_data.shape[1] // sample_frames, 50 * 32)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        i = random.randrange(1, self.raw_data.shape[1] - self.sample_frames)
        sig = self.wav[int(hparams.hop_size*(i-0.5))-1:int(hparams.hop_size*(i-0.5+self.sample_frames))]
        assert len(sig) == self.sample_frames * hparams.hop_size + 1
        prev = audio.label_2_float(sig[:-1], hparams.bits)
        return torch.Tensor(self.speaker), torch.Tensor(self.raw_data[:,i:i+self.sample_frames].T), \
                torch.Tensor(prev), torch.Tensor(sig[1:]).long()

    def __next__(self):
        i = random.randrange(0, self.raw_data.shape[1] - self.sample_frames + 1)
        return torch.Tensor(self.speaker), torch.Tensor(self.raw_data[:,i:i+self.sample_frames].T)

class SampleVideoDataset(data.Dataset):
    def __init__(self, video_path, speaker_id=0, speaker_nums=2, sample_frames=128, length=-1, ret_wav=False, use_256=False):
        super(SampleVideoDataset, self).__init__()
        data_path, video_name = os.path.split(video_path)
        folder_path = data_path + '/' + video_name.split('.')[0] + '/'
        folder = os.path.exists(folder_path)

        if not folder:  
            print("---  Creating %s...  ---" % folder_path)
            os.makedirs(folder_path)  
            reader = imageio.get_reader(video_path, 'ffmpeg', fps=20)
            for i, im in enumerate(reader):
                imageio.imwrite(folder_path + str(i).zfill(5) + '.jpg', im)
            print("---  OK  ---")
        else:
            print("---  %s already exists!  ---" % folder_path)

        self.list_frame = glob.glob(folder_path + '*.jpg')
        self.list_frame.sort()
        print("--- Totally %d video frames ---" % len(self.list_frame))

        # print(int(hparams.hop_size / hparams.sample_rate * self.sample_frames * 20))
        print('Encoding Audio of ', video_path)
        wav, sr = librosa.load(video_path, hparams.sample_rate)
        if ret_wav:
            self.wav = audio.encode_mu_law(wav, mu=2 ** hparams.bits)
        mel_basis = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels)
        linear_spec = np.abs(
            librosa.stft(wav, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size))
        mel_spec = mel_basis.dot(linear_spec)
        mel_db = 20 * np.log10(mel_spec)
        # print(in_fpath, mel_db.min(), mel_db.max())
        self.raw_data = np.clip((mel_db + 120) / 125, 0, 1)
        print('Raw_Data Shape:', self.raw_data.shape)
        if np.isnan(self.raw_data).any():
            print('!!!There exists np.nan in raw_data!!!')
        # (num_mels, num_frames)

        self.speaker = np.zeros(speaker_nums)
        self.speaker[speaker_id] = 1
        self.sample_frames = sample_frames
        if length > 0:
            self.length = length
        else:
            self.length = (max(self.raw_data.shape[1] // sample_frames, 50 * 32) // 8) * 8
        self.length = self.length // 16 * 16
        self.ret_wav = ret_wav
        self.transform = transforms.Compose([transforms.Resize(128),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if use_256:
            self.transform_large = transforms.Compose([transforms.Resize(256),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.transform_large = transforms.Compose([transforms.Resize(512),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.use_256 = use_256

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        video_length = int(hparams.hop_size / hparams.sample_rate * self.sample_frames * 20)
        # video = np.zeros((video_length, 3, 256, 256))
        video = torch.Tensor(video_length, 3, 128, 128)
        if self.use_256:
            video_large = torch.Tensor(video_length, 3, 256, 256)
        else:
            video_large = torch.Tensor(video_length, 3, 512, 512)
        if not self.ret_wav:
            i = random.randrange(0, (self.raw_data.shape[1] - self.sample_frames + 1) // 32) * 32
            video_index = int(i / 4)
            for j in range(video_length):
                video[j, :, :, :] = self.transform(Image.open(self.list_frame[j + video_index]).convert('RGB'))
                video_large[j, :, :, :] = self.transform_large(Image.open(self.list_frame[j + video_index]).convert('RGB'))
            return torch.Tensor(self.speaker), torch.Tensor(self.raw_data[:,i:i+self.sample_frames].T), video, video_large
        else:
            i = random.randrange(1, (self.raw_data.shape[1] - self.sample_frames) // 32) * 32
            sig = self.wav[int(hparams.hop_size*(i-0.5))-1:int(hparams.hop_size*(i-0.5+self.sample_frames))]
            assert len(sig) == self.sample_frames * hparams.hop_size + 1
            prev = audio.label_2_float(sig[:-1], hparams.bits)
            video_index = int(i / 4)
            for j in range(video_length):
                video[j, :, :, :] = self.transform(Image.open(self.list_frame[j + video_index]).convert('RGB'))
                video_large[j, :, :, :] = self.transform_large(Image.open(self.list_frame[j + video_index]).convert('RGB'))
            assert not (np.isnan(prev).any() or np.isnan(sig).any())
            return torch.Tensor(self.speaker), torch.Tensor(self.raw_data[:,i:i+self.sample_frames].T), \
                   torch.Tensor(prev), torch.Tensor(sig[1:]).long(), video, video_large

class MultiAudio:
    def __init__(self, data_roots, batch_size, num_workers):
        # super(OneHotDataset, self).__init__()
        self.dataloaders = [DataLoader(SampleDataset(wav_path=data_root, speaker_id=i),
                                       batch_size=batch_size, shuffle=True, num_workers=num_workers)
                            for i, data_root in enumerate(data_roots)]
        self.lengths = [len(dataloader) for dataloader in self.dataloaders]

    def __len__(self):
        return sum(self.lengths)

    def __iter__(self):
        for data in itertools.zip_longest(*self.dataloaders, fillvalue=None):
            for x in data:
                if x is not None:
                    yield x

