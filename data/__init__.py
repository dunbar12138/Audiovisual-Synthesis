import torch.nn.utils.rnn as rnn_utils
import torch
import numpy as np
from audioUtils.hparams import hparams


def collate_fn(data):
    data.sort(key=lambda x: x[1].shape[0], reverse=True)
    speakers = [x[0] for x in data]
    specs = [x[1] for x in data]
    mask = [torch.Tensor(np.ones(spec.shape, dtype=np.float)) for spec in specs]
    mask_code = [torch.Tensor(np.ones((spec.shape[0]//hparams.freq, hparams.dim_neck*2), dtype=np.float)) for spec in specs]
    data_length = [spec.shape[0] for spec in specs]
    specs = rnn_utils.pad_sequence(specs, batch_first=True, padding_value=0)
    speakers = rnn_utils.pad_sequence(speakers, batch_first=True, padding_value=0)
    mask = rnn_utils.pad_sequence(mask, batch_first=True, padding_value=0)
    mask_code = rnn_utils.pad_sequence(mask_code, batch_first=True, padding_value=0)
    return speakers, specs, data_length, mask, mask_code, [x[2] for x in data]
