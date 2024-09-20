import torch 
import numpy as np 
import torchaudio

class DataProcessor():
    def __init__(self, mean_std_pth, device, fp):
        mean, std = self.load_mean_std(mean_std_pth)
        self.mean = mean.to(device, dtype=torch.float32)
        self.std = std.to(device, dtype=torch.float32)
        self.fp = fp 
        self.device = device

    def load_mean_std(self, mean_std_npy_path):
        mean_std = np.load(mean_std_npy_path)
        mean = torch.Tensor(mean_std[0].reshape(-1))
        std = torch.Tensor(mean_std[1].reshape(-1))
        return mean, std  

    def extract_fbank(self, waveform_path):
        waveform, sr = torchaudio.load(waveform_path)
        waveform = waveform*(2**15)
        y = torchaudio.compliance.kaldi.fbank(
                            waveform,
                            num_mel_bins=40,
                            sample_frequency=16000,
                            window_type='hamming',
                            frame_length=25,
                            frame_shift=10).to(self.device)
        # Normalize by the mean and std of Librispeech
        y = (y-self.mean)/self.std
        # Downsampling by twice 
        if self.fp == 20:
            odd_y = y[::2,:]
            even_y = y[1::2,:]
            if odd_y.shape[0] != even_y.shape[0]:
                even_y = torch.cat((even_y, torch.zeros(1,even_y.shape[1]).to(self.device)), dim=0)
            y = torch.cat((odd_y, even_y), dim=1)
        return y
    
    def prepare_data(self, waveform_path):
        mel_input = self.extract_fbank(waveform_path).unsqueeze(0)
        pad_mask = torch.ones(mel_input.shape[:-1]).to(self.device)
        return mel_input, pad_mask