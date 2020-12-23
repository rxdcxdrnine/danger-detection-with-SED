import random
import glob
from tqdm import tqdm
import random
import numpy as np
import torch
import torchaudio

from scipy.io.wavfile import write


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, args, train=True):
        if train:
            self.scene_filepath = args["train_filepath"]
            self.event_filepath = args["train_event_filepath"]
        else:
            self.scene_filepath = args["test_filepath"]
            self.event_filepath = args["test_event_filepath"]
        
        self.samp_rate = args["samp_rate"]
        self.nb_samp = int(args["samp_rate"] * 0.001 * args["utt_length"])
        self.win_length = args["win_length"]
        self.hop_length = args["hop_length"]
        self.n_fft = args["n_fft"]
        self.n_mels = args["n_mels"]
        
    def __len__(self):
        return len(self.scene_filepath)
    
    def _cut(self, X):
        if X.size(0) < self.nb_samp:
            nb_dup = int(np.ceil(self.nb_samp / X.size(0)))
            X = X.repeat(nb_dup)
        
        margin = X.size(0) - self.nb_samp
        st_idx = random.randint(0, margin)
        X = X[st_idx:(st_idx + self.nb_samp)]

        return X
    
    def _mix_audio(self, scene_audio):
        index = random.randint(0, len(self.event_filepath) - 1)
        
        event = self.event_filepath[index]
        event_audio, _  = torchaudio.load(event, normalization=True)
        event_audio = event_audio.squeeze(0)
        event_expand = torch.zeros(scene_audio.size(0))
        
        margin = event_expand.size(0) - event_audio.size(0)
        st_idx = random.randint(0, margin)
        event_expand[st_idx:(st_idx + event_audio.size(0))] = event_audio
        
        result = scene_audio + event_expand   # scene_audio amplitude downsize
        return result

    def _local_cep_mvn(self, X):
        X_mean = X.mean(dim=-1, keepdim=True)
        X_std = X.std(dim=-1, keepdim=True)
        X_std[X_std < .001] = .001
        result = (X - X_mean) / X_std
        
        return result
    
    def __getitem__(self, index):
        scene = self.scene_filepath[index]
        # scene = self.scene_filepath[3000]     # for demo test synthetic file
        scene_audio, _ = torchaudio.load(scene, normalization=True)   #X.size : (1, 441000)
        scene_audio = scene_audio.squeeze(0)
        

        X = self._cut(scene_audio)
        label = random.choices([0, 1], weights=[0.3, 0.7])[0]
        if label: X = self._mix_audio(X)
          
        log_mel_spec_extractor = torchaudio.transforms.MelSpectrogram(
            self.samp_rate,
            n_fft=self.n_fft,
            win_length=int(self.samp_rate * 0.001 * self.win_length),
            hop_length=int(self.samp_rate * 0.001 * self.hop_length),
            window_fn=torch.hamming_window,
            n_mels=self.n_mels
        )
        
        X = log_mel_spec_extractor(X)
        X = torch.log(X)
        X = self._local_cep_mvn(X)
        
        X = X.unsqueeze(0)
        
        return X, label
