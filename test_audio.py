import argparse

from model_vc import Generator
import torch
from audioUtils import audio
from audioUtils.hparams import hparams
from audioUtils.audio import preemphasis, inv_preemphasis
import librosa
from pathlib import Path
import numpy as np
from math import ceil
import glob


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0]) / base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0, len_pad), (0, 0)), 'constant'), len_pad


mel_basis = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=80)

def voice_conversion(G, input_wavfile, parallel=True):
    source_path = input_wavfile
    wav, sr = librosa.load(source_path, hparams.sample_rate)
    wav = preemphasis(wav, hparams.preemphasis, hparams.sample_rate)
    linear_spec = np.abs(
        librosa.stft(wav, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size))
    mel_spec = mel_basis.dot(linear_spec)
    mel_db = 20 * np.log10(mel_spec)
    source_spec = np.clip((mel_db + 120) / 125, 0, 1)
    # source_embed = torch.from_numpy(np.array([0, 1])).float()

    source_spec, _ = pad_seq(source_spec.T, hparams.freq)

    with torch.no_grad():
        s2t_spec = G.conversion(torch.Tensor(source_spec).unsqueeze(0), device).cpu()

    if parallel:
        s2t_wav = G.vocoder.generate(s2t_spec.transpose(1, 2), True, 8000, 800, mu_law=True)
    else:
        s2t_wav = G.vocoder.generate(s2t_spec.transpose(1, 2), False, None, None, mu_law=True)

    s2t_wav = inv_preemphasis(s2t_wav, hparams.preemphasis, hparams.preemphasize)
    librosa.output.write_wav(args.output_file, s2t_wav.astype(np.float32), hparams.sample_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_path')
    parser.add_argument('--model')
    parser.add_argument('--parallel', dest='parallel', default=False, action='store_true')
    parser.add_argument('--output_file')
    args = parser.parse_args()

    device = "cuda:0"

    model_path = args.model

    G = Generator(hparams.dim_neck, hparams.speaker_embedding_size, 512, hparams.freq, is_train=False,
                  discriminator=False).to(device)

    print("Loading autovc model...", end='\t')
    load_model = model_path
    d = torch.load(load_model)
    newdict = d.copy()
    for key, value in d.items():
        newkey = key
        if 'wavenet' in key:
            newdict[key.replace('wavenet', 'vocoder')] = newdict.pop(key)
            newkey = key.replace('wavenet', 'vocoder')
        if 'module' in key:
            newdict[newkey.replace('module.','',1)] = newdict.pop(newkey)
            newkey = newkey.replace('module.', '', 1)
        if newkey not in G.state_dict():
            #print(newkey)
            newdict.pop(newkey)
    print("Load " + str(len(newdict)) + " parameters!")
    G.load_state_dict(newdict, strict=False)
    print("Done.")

    voice_conversion(G, args.wav_path, args.parallel)
