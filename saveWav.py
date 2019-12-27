import cmath

import numpy as np

import librosa
from audioUtils.hparams import hparams
from audioUtils.audio import _griffin_lim, seg2wav, inv_preemphasis
import os

_inv_mel_basis = np.linalg.pinv(librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=80))

def saveWav(mel, fpath):
    mel = mel * 8 - 4
    generated_wav = vocoder.infer_waveform(mel)
    librosa.output.write_wav(fpath, generated_wav.astype(np.float32),
                             hparams.sample_rate)


def mel2wav(mel):
    mel = np.clip(mel, 0, 1)
    mel = mel * 125 - 120
    mel = np.power(10.0, mel * 0.05)
    spec = _inv_mel_basis.dot(mel)
    generated_wav = _griffin_lim(spec, hparams)

    generated_wav = inv_preemphasis(generated_wav, hparams.preemphasis, hparams.preemphasize)
    return generated_wav, hparams.sample_rate


if __name__ == "__main__":
    import pickle
    # f = open("/home/kangled/datasets/VCTK/VCTK-Corpus/mydata/p229_p229_001.pkl", "rb")
    # data = pickle.load(f)
    # f.close()
    # saveWav(data['spec'], "/home/kangled/test.wav")
    a = np.load("/scratch/kangled/autovc/Epoch[003]p261_p261_184p343.npy")
    saveWav(a.T, "/home/kangled/test.wav")
