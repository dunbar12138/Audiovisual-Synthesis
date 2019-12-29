from pathlib import Path

import imageio
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

from scipy.fftpack import dct
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from audioUtils.audio import wav2seg, inv_preemphasis, preemphasis
from data.Sample_dataset import pad_seq
from model_video import VideoGenerator, STAGE2_G
from saveWav import mel2wav
from audioUtils.hparams import hparams
from audioUtils import audio
from vocoder.models.fatchord_version import WaveRNN
import cv2

_inv_mel_basis = np.linalg.pinv(audio._build_mel_basis(hparams))
mel_basis = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=40)

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class MyEncoder(nn.Module):
    '''Encoder without speaker embedding'''

    def __init__(self, dim_neck, freq, num_mel=80):
        super(MyEncoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(num_mel if i == 0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, c_org=None, return_unsample=False):
        # (B, T, n_mel)
        x = x.squeeze(1).transpose(2, 1)

        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]

        # print(outputs.shape)

        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:, i + self.freq - 1, :], out_backward[:, i, :]), dim=-1))
        if return_unsample:
            return codes, outputs
        return codes


class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_emb, dim_pre, num_mel=80):
        super(Decoder, self).__init__()
        
        self.lstm1 = nn.LSTM(dim_neck*2+dim_emb, dim_pre, 1, batch_first=True)
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        
        self.linear_projection = LinearNorm(1024, num_mel)

    def forward(self, x):
        
        #self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        outputs, _ = self.lstm2(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output   

    
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, num_mel=80):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(num_mel, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, num_mel,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(num_mel))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


def pad_layer(inp, layer, is_2d=False):
    if type(layer.kernel_size) == tuple:
        kernel_size = layer.kernel_size[0]
    else:
        kernel_size = layer.kernel_size
    if not is_2d:
        if kernel_size % 2 == 0:
            pad = (kernel_size//2, kernel_size//2 - 1)
        else:
            pad = (kernel_size//2, kernel_size//2)
    else:
        if kernel_size % 2 == 0:
            pad = (kernel_size//2, kernel_size//2 - 1, kernel_size//2, kernel_size//2 - 1)
        else:
            pad = (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2)
    # padding
    inp = F.pad(inp,
            pad=pad,
            mode='reflect')
    out = layer(inp)
    return out


class PatchDiscriminator(nn.Module):
    def __init__(self, n_class=33, ns=0.2, dp=0.1):
        super(PatchDiscriminator, self).__init__()
        self.ns = ns
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2)
        self.conv6 = nn.Conv2d(512, 1, kernel_size=1)
        #self.conv_classify = nn.Conv2d(512, n_class, kernel_size=(17, 4))
        # self.conv_classify = nn.Conv2d(512, n_class, kernel_size=(17, 4))
        self.drop1 = nn.Dropout2d(p=dp)
        self.drop2 = nn.Dropout2d(p=dp)
        self.drop3 = nn.Dropout2d(p=dp)
        self.drop4 = nn.Dropout2d(p=dp)
        self.drop5 = nn.Dropout2d(p=dp)
        self.ins_norm1 = nn.InstanceNorm2d(self.conv1.out_channels)
        self.ins_norm2 = nn.InstanceNorm2d(self.conv2.out_channels)
        self.ins_norm3 = nn.InstanceNorm2d(self.conv3.out_channels)
        self.ins_norm4 = nn.InstanceNorm2d(self.conv4.out_channels)
        self.ins_norm5 = nn.InstanceNorm2d(self.conv5.out_channels)

    def conv_block(self, x, conv_layer, after_layers):
        out = pad_layer(x, conv_layer, is_2d=True)
        out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in after_layers:
            out = layer(out)
        return out

    def forward(self, x, classify=False):
        x = torch.unsqueeze(x, dim=1)
        out = self.conv_block(x, self.conv1, [self.ins_norm1, self.drop1])
        out = self.conv_block(out, self.conv2, [self.ins_norm2, self.drop2])
        out = self.conv_block(out, self.conv3, [self.ins_norm3, self.drop3])
        out = self.conv_block(out, self.conv4, [self.ins_norm4, self.drop4])
        out = self.conv_block(out, self.conv5, [self.ins_norm5, self.drop5])
        # GAN output value
        val = pad_layer(out, self.conv6, is_2d=True)
        val = val.view(val.size(0), -1)
        mean_val = torch.mean(val, dim=1)
        # if classify:
        #     # classify
        #     logits = self.conv_classify(out)
        #     print(logits.size())
        #     logits = logits.view(logits.size(0), -1)
        #     return mean_val, logits
        # else:
        return mean_val


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq, dim_spec=80, is_train=False, lr=0.001, loss_content=True,
                 discriminator=False, multigpu=False, lambda_gan=0.0001,
                 lambda_wavenet=0.001, args=None,
                 test_path_source=None, test_path_target=None):
        super(Generator, self).__init__()

        self.encoder = MyEncoder(dim_neck, freq, num_mel=dim_spec)
        self.decoder = Decoder(dim_neck, 0, dim_pre, num_mel=dim_spec)
        self.postnet = Postnet(num_mel=dim_spec)

        if discriminator:
            self.dis = PatchDiscriminator(n_class=num_speakers)
            self.dis_criterion = GANLoss(use_lsgan=use_lsgan, tensor=torch.cuda.FloatTensor)
        else:
            self.dis = None

        self.loss_content = loss_content
        self.lambda_gan = lambda_gan
        self.lambda_wavenet = lambda_wavenet

        self.multigpu = multigpu
        self.prepare_test(dim_spec, test_path_source, test_path_target)

        self.vocoder = WaveRNN(
            rnn_dims=hparams.voc_rnn_dims,
            fc_dims=hparams.voc_fc_dims,
            bits=hparams.bits,
            pad=hparams.voc_pad,
            upsample_factors=hparams.voc_upsample_factors,
            feat_dims=hparams.num_mels,
            compute_dims=hparams.voc_compute_dims,
            res_out_dims=hparams.voc_res_out_dims,
            res_blocks=hparams.voc_res_blocks,
            hop_length=hparams.hop_size,
            sample_rate=hparams.sample_rate,
            mode=hparams.voc_mode
        )
        
        if is_train:
            self.criterionIdt = torch.nn.L1Loss(reduction='mean')
            self.opt_encoder = torch.optim.Adam(self.encoder.parameters(), lr=lr)
            self.opt_decoder = torch.optim.Adam(itertools.chain(self.decoder.parameters(), self.postnet.parameters()), lr=lr)
            if discriminator:
                self.opt_dis = torch.optim.Adam(self.dis.parameters(), lr=lr)
            self.opt_vocoder = torch.optim.Adam(self.vocoder.parameters(), lr=hparams.voc_lr)
            self.vocoder_loss_func = F.cross_entropy # Only for RAW


        if multigpu:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)
            self.postnet = nn.DataParallel(self.postnet)
            self.vocoder = nn.DataParallel(self.vocoder)
            if self.dis is not None:
                self.dis = nn.DataParallel(self.dis)

    def prepare_test(self, dim_spec, source_path=None, target_path=None):
        if source_path is None:
            source_path = "/mnt/lustre/dengkangle/cmu/datasets/audio/test/trump_02.wav"
        if target_path is None:
            target_path = "/mnt/lustre/dengkangle/cmu/datasets/audio/test/female.wav"
        # source_path = "/home/kangled/datasets/audio/Chaplin_01.wav"
        # target_path = "/home/kangled/datasets/audio/Obama_01.wav"

        mel_basis80 = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=80)

        wav, sr = librosa.load(source_path, hparams.sample_rate)
        wav = preemphasis(wav, hparams.preemphasis, hparams.preemphasize)
        linear_spec = np.abs(
            librosa.stft(wav, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size))
        mel_spec = mel_basis80.dot(linear_spec)
        mel_db = 20 * np.log10(mel_spec)
        source_spec = np.clip((mel_db + 120) / 125, 0, 1)
        # source_spec = mel_spec

        self.source_embed = torch.from_numpy(np.array([0, 1])).float().unsqueeze(0)
        self.source_wav = wav

        wav, sr = librosa.load(target_path, hparams.sample_rate)
        wav = preemphasis(wav, hparams.preemphasis, hparams.preemphasize)
        linear_spec = np.abs(
            librosa.stft(wav, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size))
        mel_spec = mel_basis80.dot(linear_spec)
        mel_db = 20 * np.log10(mel_spec)
        target_spec = np.clip((mel_db + 120) / 125, 0, 1)
        # target_spec = mel_spec
        
        self.target_embed = torch.from_numpy(np.array([1, 0])).float().unsqueeze(0)
        self.target_wav = wav

        self.source_spec = torch.Tensor(pad_seq(source_spec.T, hparams.freq)).unsqueeze(0)
        self.target_spec = torch.Tensor(pad_seq(target_spec.T, hparams.freq)).unsqueeze(0)

    def test_fixed(self, device):
        with torch.no_grad():
            t2s_spec = self.conversion(self.target_embed, self.source_embed, self.target_spec, device).cpu()
            s2s_spec = self.conversion(self.source_embed, self.source_embed, self.source_spec, device).cpu()
            s2t_spec = self.conversion(self.source_embed, self.target_embed, self.source_spec, device).cpu()
            t2t_spec = self.conversion(self.target_embed, self.target_embed, self.target_spec, device).cpu()

        ret_dic = {}
        ret_dic['A_fake_griffin'], sr = mel2wav(s2t_spec.numpy().squeeze(0).T)
        ret_dic['B_fake_griffin'], sr = mel2wav(t2s_spec.numpy().squeeze(0).T)
        ret_dic['A'] = self.source_wav
        ret_dic['B'] = self.target_wav

        with torch.no_grad():
            if not self.multigpu:
                ret_dic['A_fake_w'] = inv_preemphasis(self.vocoder.generate(s2t_spec.to(device).transpose(2, 1), False, None, None, mu_law=True),
                                                hparams.preemphasis, hparams.preemphasize)
                ret_dic['B_fake_w'] = inv_preemphasis(self.vocoder.generate(t2s_spec.to(device).transpose(2, 1), False, None, None, mu_law=True),
                                                hparams.preemphasis, hparams.preemphasize)
            else:
                ret_dic['A_fake_w'] = inv_preemphasis(self.vocoder.module.generate(s2t_spec.to(device).transpose(2, 1), False, None, None, mu_law=True),
                                                hparams.preemphasis, hparams.preemphasize)
                ret_dic['B_fake_w'] = inv_preemphasis(self.vocoder.module.generate(t2s_spec.to(device).transpose(2, 1), False, None, None, mu_law=True),
                                                hparams.preemphasis, hparams.preemphasize)
        return ret_dic, sr


    def conversion(self, speaker_org, speaker_trg, spec, device, speed=1):
        speaker_org, speaker_trg, spec = speaker_org.to(device), speaker_trg.to(device), spec.to(device)
        if not self.multigpu:
            codes = self.encoder(spec, speaker_org)
        else:
            codes = self.encoder.module(spec, speaker_org)
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1, int(speed * spec.size(1) / len(codes)), -1))
        code_exp = torch.cat(tmp, dim=1)
        encoder_outputs = torch.cat((code_exp, speaker_trg.unsqueeze(1).expand(-1, code_exp.size(1), -1)), dim=-1)
        mel_outputs = self.decoder(code_exp) if not self.multigpu else self.decoder.module(code_exp)

        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)
        return mel_outputs_postnet

    def optimize_parameters(self, dataloader, epochs, device, display_freq=10, save_freq=1000, save_dir="./",
                            experimentName="Train", load_model=None, initial_niter=0):
        writer = SummaryWriter(log_dir="logs/"+experimentName)
        if load_model is not None:
            print("Loading from %s..." % load_model)
            # self.load_state_dict(torch.load(load_model))
            d = torch.load(load_model)
            newdict = d.copy()
            for key, value in d.items():
                newkey = key
                if 'wavenet' in key:
                    newdict[key.replace('wavenet', 'vocoder')] = newdict.pop(key)
                    newkey = key.replace('wavenet', 'vocoder')
                if self.multigpu and 'module' not in key:
                    newdict[newkey.replace('.','.module.',1)] = newdict.pop(newkey)
                    newkey = newkey.replace('.', '.module.', 1)
                if newkey not in self.state_dict():
                    newdict.pop(newkey)
            self.load_state_dict(newdict)
            print("AutoVC Model Loaded")
        niter = initial_niter
        for epoch in range(epochs):
            self.train()
            for i, data in enumerate(dataloader):
                speaker_org, spec, prev, wav = data
                loss_dict, loss_dict_discriminator, loss_dict_wavenet = \
                    self.train_step(spec.to(device), speaker_org.to(device), prev=prev.to(device), wav=wav.to(device), device=device)
                if niter % display_freq == 0:
                    print("Epoch[%d] Iter[%d] Niter[%d] %s %s %s"
                          % (epoch, i, niter, loss_dict, loss_dict_discriminator, loss_dict_wavenet))
                    writer.add_scalars('data/Loss', loss_dict,
                                       niter)
                    if loss_dict_discriminator != {}:
                        writer.add_scalars('data/discriminator', loss_dict_discriminator, niter)
                    if loss_dict_wavenet != {}:
                        writer.add_scalars('data/wavenet', loss_dict_wavenet, niter)
                if niter % save_freq == 0:
                    print("Saving and Testing...", end='\t')
                    torch.save(self.state_dict(), save_dir + '/Epoch' + str(epoch).zfill(3) + '_Iter'
                               + str(niter).zfill(8) + ".pkl")
                    # self.load_state_dict(torch.load('params.pkl'))
                    if len(dataloader) >= 2:
                        wav_dic, sr = self.test_fixed(device)
                        for key, wav in wav_dic.items():
                            # print(wav.shape)
                            writer.add_audio(key, wav, niter, sample_rate=sr)
                    print("Done")
                    self.train()
                torch.cuda.empty_cache()  # Prevent Out of Memory
                niter += 1


    def train_step(self, x, c_org, mask=None, mask_code=None, prev=None, wav=None,
                   ret_content=False, retain_graph=False, device='cuda:0'):
        codes = self.encoder(x, c_org)
        # print(codes[0].shape)
        content = torch.cat([code.unsqueeze(1) for code in codes], dim=1)
        # print("content shape", content.shape)
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1, int(x.size(1) / len(codes)), -1))
        code_exp = torch.cat(tmp, dim=1)

        encoder_outputs = torch.cat((code_exp, c_org.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)

        mel_outputs = self.decoder(code_exp)

        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)

        loss_dict, loss_dict_discriminator, loss_dict_wavenet = {}, {}, {}

        loss_recon = self.criterionIdt(x, mel_outputs)
        loss_recon0 = self.criterionIdt(x, mel_outputs_postnet)
        loss_dict['recon'], loss_dict['recon0'] = loss_recon.data.item(), loss_recon0.data.item()

        if self.loss_content:
            recons_codes = self.encoder(mel_outputs_postnet, c_org)
            recons_content = torch.cat([code.unsqueeze(1) for code in recons_codes], dim=1)
            if mask is not None:
                loss_content = self.criterionIdt(content.masked_select(mask_code.byte()), recons_content.masked_select(mask_code.byte()))
            else:
                loss_content = self.criterionIdt(content, recons_content)
            loss_dict['content'] = loss_content.data.item()
        else:
            loss_content = torch.from_numpy(np.array(0))

        loss_gen, loss_dis, loss_vocoder = [torch.from_numpy(np.array(0))] * 3
        fake_mel = None
        if self.dis:
            # true_label = torch.from_numpy(np.ones(shape=(x.shape[0]))).to('cuda:0').long()
            # false_label = torch.from_numpy(np.zeros(shape=(x.shape[0]))).to('cuda:0').long()

            flip_speaker = 1 - c_org
            fake_mel = self.conversion(c_org, flip_speaker, x, device)

            loss_dis = self.dis_criterion(self.dis(x), True) + self.dis_criterion(self.dis(fake_mel), False)
                       # +  self.dis_criterion(self.dis(mel_outputs_postnet), False)

            self.opt_dis.zero_grad()
            loss_dis.backward(retain_graph=True)
            self.opt_dis.step()
            loss_gen = self.dis_criterion(self.dis(fake_mel), True)
                # + self.dis_criterion(self.dis(mel_outputs_postnet), True)
            loss_dict_discriminator['dis'], loss_dict_discriminator['gen'] = loss_dis.data.item(), loss_gen.data.item()


        if not self.multigpu:
            y_hat = self.vocoder(prev,
                                self.vocoder.pad_tensor(mel_outputs_postnet, hparams.voc_pad).transpose(1, 2))
        else:
            y_hat = self.vocoder(prev,self.vocoder.module.pad_tensor(mel_outputs_postnet, hparams.voc_pad).transpose(1, 2))
        y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
        # assert (0 <= wav < 2 ** 9).all()
        loss_vocoder = self.vocoder_loss_func(y_hat, wav.unsqueeze(-1).to(device))
        self.opt_vocoder.zero_grad()

        Loss = loss_recon + loss_recon0 + loss_content + \
               self.lambda_gan * loss_gen + self.lambda_wavenet * loss_vocoder
        loss_dict['total'] = Loss.data.item()
        self.opt_encoder.zero_grad()
        self.opt_decoder.zero_grad()
        Loss.backward(retain_graph=retain_graph)
        self.opt_encoder.step()
        self.opt_decoder.step()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.vocoder.parameters(), 65504.0)
        self.opt_vocoder.step()

        if ret_content:
            return loss_recon, loss_recon0, loss_content, Loss, content
        return loss_dict, loss_dict_discriminator, loss_dict_wavenet
   

class VideoAudioGenerator(nn.Module):
    def __init__(self, dim_neck, dim_emb, dim_pre, freq, dim_spec=80, is_train=False, lr=0.001,
                 multigpu=False, 
                 lambda_wavenet=0.001, args=None,
                 residual=False, attention_map=None, use_256=False, loss_content=False,
                 test_path=None):
        super(VideoAudioGenerator, self).__init__()

        self.encoder = MyEncoder(dim_neck, freq, num_mel=dim_spec)

        self.decoder = Decoder(dim_neck, 0, dim_pre, num_mel=dim_spec)
        self.postnet = Postnet(num_mel=dim_spec)
        if use_256:
            self.video_decoder = VideoGenerator(use_256=True)
        else:
            self.video_decoder = STAGE2_G(residual=residual)
        self.use_256 = use_256
        self.lambda_wavenet = lambda_wavenet
        self.loss_content = loss_content
        self.multigpu = multigpu
        self.test_path = test_path

        self.vocoder = WaveRNN(
            rnn_dims=hparams.voc_rnn_dims,
            fc_dims=hparams.voc_fc_dims,
            bits=hparams.bits,
            pad=hparams.voc_pad,
            upsample_factors=hparams.voc_upsample_factors,
            feat_dims=hparams.num_mels,
            compute_dims=hparams.voc_compute_dims,
            res_out_dims=hparams.voc_res_out_dims,
            res_blocks=hparams.voc_res_blocks,
            hop_length=hparams.hop_size,
            sample_rate=hparams.sample_rate,
            mode=hparams.voc_mode
        )

        if is_train:
            self.criterionIdt = torch.nn.L1Loss(reduction='mean')
            self.opt_encoder = torch.optim.Adam(self.encoder.parameters(), lr=lr)
            self.opt_decoder = torch.optim.Adam(itertools.chain(self.decoder.parameters(), self.postnet.parameters()), lr=lr)
            self.opt_video_decoder = torch.optim.Adam(self.video_decoder.parameters(), lr=lr)

            self.opt_vocoder = torch.optim.Adam(self.vocoder.parameters(), lr=hparams.voc_lr)
            self.vocoder_loss_func = F.cross_entropy # Only for RAW

        if multigpu:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)
            self.video_decoder = nn.DataParallel(self.video_decoder)
            self.postnet = nn.DataParallel(self.postnet)
            self.vocoder = nn.DataParallel(self.vocoder)
    
    def optimize_parameters_video(self, dataloader, epochs, device, display_freq=10, save_freq=1000, save_dir="./",
                            experimentName="Train", initial_niter=0, load_model=None):
        writer = SummaryWriter(log_dir="logs/" + experimentName)
        if load_model is not None:
            print("Loading from %s..." % load_model)
            # self.load_state_dict(torch.load(load_model))
            d = torch.load(load_model)
            newdict = d.copy()
            for key, value in d.items():
                newkey = key
                if 'wavenet' in key:
                    newdict[key.replace('wavenet', 'vocoder')] = newdict.pop(key)
                    newkey = key.replace('wavenet', 'vocoder')
                if self.multigpu and 'module' not in key:
                    newdict[newkey.replace('.','.module.',1)] = newdict.pop(newkey)
                    newkey = newkey.replace('.', '.module.', 1)
                if newkey not in self.state_dict():
                    newdict.pop(newkey)
            print("Load " + str(len(newdict)) + " parameters!")
            self.load_state_dict(newdict, strict=False)
            print("AutoVC Model Loaded") 
        niter = initial_niter
        for epoch in range(epochs):
            self.train()
            for i, data in enumerate(dataloader):
                # print("Processing ..." + str(name))
                speaker, mel, prev, wav, video, video_large = data
                speaker, mel, prev, wav, video, video_large = speaker.to(device), mel.to(device), prev.to(device), wav.to(device), video.to(device), video_large.to(device)
                codes, code_unsample = self.encoder(mel, speaker, return_unsample=True)
                
                tmp = []
                for code in codes:
                    tmp.append(code.unsqueeze(1).expand(-1, int(mel.size(1) / len(codes)), -1))
                code_exp = torch.cat(tmp, dim=1)

                if not self.use_256:
                    v_stage1, v_stage2 = self.video_decoder(code_unsample, train=True)
                else:
                    v_stage2 = self.video_decoder(code_unsample)
                mel_outputs = self.decoder(code_exp)
                mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
                mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)

                if self.loss_content:
                    _, recons_codes = self.encoder(mel_outputs_postnet, speaker, return_unsample=True)
                    loss_content = self.criterionIdt(code_unsample, recons_codes)
                else:
                    loss_content = torch.from_numpy(np.array(0))
                
                if not self.use_256:
                    loss_video = self.criterionIdt(v_stage1, video) + self.criterionIdt(v_stage2, video_large)
                else:
                    loss_video = self.criterionIdt(v_stage2, video_large)
                
                loss_recon = self.criterionIdt(mel, mel_outputs)
                loss_recon0 = self.criterionIdt(mel, mel_outputs_postnet)
                loss_vocoder = 0

                if not self.multigpu:
                    y_hat = self.vocoder(prev,
                                    self.vocoder.pad_tensor(mel_outputs_postnet, hparams.voc_pad).transpose(1, 2))
                else:
                    y_hat = self.vocoder(prev,self.vocoder.module.pad_tensor(mel_outputs_postnet, hparams.voc_pad).transpose(1, 2))
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
                # assert (0 <= wav < 2 ** 9).all()
                loss_vocoder = self.vocoder_loss_func(y_hat, wav.unsqueeze(-1).to(device))
                self.opt_vocoder.zero_grad()

                loss = loss_video + loss_recon + loss_recon0 + self.lambda_wavenet * loss_vocoder + loss_content

                self.opt_encoder.zero_grad()
                self.opt_decoder.zero_grad()
                self.opt_video_decoder.zero_grad()
                loss.backward()
                self.opt_encoder.step()
                self.opt_decoder.step()
                self.opt_video_decoder.step()
                self.opt_vocoder.step()



                if niter % display_freq == 0:
                    print("Epoch[%d] Iter[%d] Niter[%d] %s"
                          % (epoch, i, niter, loss.data.item()))
                    writer.add_scalars('data/Loss', {'loss':loss.data.item(),
                                                    'loss_video':loss_video.data.item(),
                                                    'loss_audio':loss_recon0.data.item()+loss_recon.data.item()}, niter)

                if niter % save_freq == 0:
                    torch.cuda.empty_cache()  # Prevent Out of Memory
                    print("Saving and Testing...", end='\t')
                    torch.save(self.state_dict(), save_dir + '/Epoch' + str(epoch).zfill(3) + '_Iter'
                               + str(niter).zfill(8) + ".pkl")
                    # self.load_state_dict(torch.load('params.pkl'))
                    self.test_audiovideo(device, writer, niter)
                    print("Done")
                    self.train()
                torch.cuda.empty_cache()  # Prevent Out of Memory
                niter += 1

    def generate(self, mel, speaker, device='cuda:0'):
        mel, speaker = mel.to(device), speaker.to(device)
        if not self.multigpu:
            codes, code_unsample = self.encoder(mel, speaker, return_unsample=True)
        else:
            codes, code_unsample = self.encoder.module(mel, speaker, return_unsample=True)
                
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1, int(mel.size(1) / len(codes)), -1))
        code_exp = torch.cat(tmp, dim=1)

        if not self.multigpu:
            if not self.use_256:
                v_stage1, v_stage2 = self.video_decoder(code_unsample, train=True)
            else:
                v_stage2 = self.video_decoder(code_unsample)
                v_stage1 = v_stage2
            mel_outputs = self.decoder(code_exp)
            mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
        else:
            if not self.use_256:
                v_stage1, v_stage2 = self.video_decoder.module(code_unsample, train=True)
            else:
                v_stage2 = self.video_decoder.module(code_unsample)
                v_stage1 = v_stage2
            mel_outputs = self.decoder.module(code_exp)
            mel_outputs_postnet = self.postnet.module(mel_outputs.transpose(2, 1))
        
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)
        
        return mel_outputs_postnet, v_stage1, v_stage2
    
    def test_video(self, device):
        wav, sr = librosa.load("/mnt/lustre/dengkangle/cmu/datasets/video/obama_test.mp4", hparams.sample_rate)
        mel_basis = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels)
        linear_spec = np.abs(
            librosa.stft(wav, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size))
        mel_spec = mel_basis.dot(linear_spec)
        mel_db = 20 * np.log10(mel_spec)

        test_data = np.clip((mel_db + 120) / 125, 0, 1)
        test_data = torch.Tensor(pad_seq(test_data.T, hparams.freq)).unsqueeze(0).to(device)
        with torch.no_grad():
            codes, code_exp = self.encoder.module(test_data, return_unsample=True)
            v_mid, v_hat = self.video_decoder.module(code_exp, train=True)

        reader = imageio.get_reader("/mnt/lustre/dengkangle/cmu/datasets/video/obama_test.mp4", 'ffmpeg', fps=20)
        frames = []
        for i, im in enumerate(reader):
            frames.append(np.array(im).transpose(2, 0, 1))
        frames = (np.array(frames) / 255 - 0.5) / 0.5
        return frames, v_mid[0:1], v_hat[0:1]

    def test_audiovideo(self, device, writer, niter):
        source_path = self.test_path

        mel_basis80 = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=80)

        wav, sr = librosa.load(source_path, hparams.sample_rate)
        wav = preemphasis(wav, hparams.preemphasis, hparams.preemphasize)

        linear_spec = np.abs(
            librosa.stft(wav, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size))
        mel_spec = mel_basis80.dot(linear_spec)
        mel_db = 20 * np.log10(mel_spec)
        source_spec = np.clip((mel_db + 120) / 125, 0, 1)
        
        source_embed = torch.from_numpy(np.array([0, 1])).float().unsqueeze(0)
        source_wav = wav

        source_spec = torch.Tensor(pad_seq(source_spec.T, hparams.freq)).unsqueeze(0)
        # print(source_spec.shape)
        
        with torch.no_grad():
            generated_spec, v_mid, v_hat = self.generate(source_spec, source_embed ,device)

        generated_spec, v_mid, v_hat = generated_spec.cpu(), v_mid.cpu(), v_hat.cpu()

        print("Generating Wavfile...")
        with torch.no_grad():
            if not self.multigpu:
                generated_wav = inv_preemphasis(self.vocoder.generate(generated_spec.to(device).transpose(2, 1), False, None, None, mu_law=True), hparams.preemphasis, hparams.preemphasize)
            
            else:
                generated_wav = inv_preemphasis(self.vocoder.module.generate(generated_spec.to(device).transpose(2, 1), False, None, None, mu_law=True), hparams.preemphasis, hparams.preemphasize)


        writer.add_video('generated', (v_hat.numpy()+1)/2, global_step=niter)
        writer.add_video('mid', (v_mid.numpy()+1)/2, global_step=niter)
        writer.add_audio('ground_truth', source_wav, niter, sample_rate=hparams.sample_rate)
        writer.add_audio('generated_wav', generated_wav, niter, sample_rate=hparams.sample_rate)
