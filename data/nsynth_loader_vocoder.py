# External Libraries
import os
import json
import glob
import numpy as np
import scipy.io.wavfile
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import librosa
import random
import math

# Internal Libraries
import lib.signal_utils as sign_op
from lib.normalizer import DataNormalizer
from lib.layers import TacotronSTFT

MAX_WAV_VALUE = 32768.0
MAX_INDEX_WAV = 2*22050 #2 seconds at 22KHz rate.

def expand(mat):
    while not math.log(mat.shape[1], 2).is_integer():
        expand_vec = np.expand_dims(mat[:,-1],axis=1)
        mat = np.hstack((mat,expand_vec))
    return mat

class NSynthVocoder(data.Dataset):

    """Pytorch dataset for NSynth Wav datasetinst_list
    args:
        root: root dir containing examples.json and audio directory with
            wav files.
        transform (callable, optional): A function/transform that takes in
                a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        blacklist_pattern: list of string used to blacklist dataset element.
            If one of the string is present in the audio filename, this sample
            together with its metadata is removed from the dataset.
        categorical_field_list: list of string. Each string is a key like
            instrument_family that will be used as a classification target.
            Each field value will be encoding as an integer using sklearn
            LabelEncoder.
    """

    def __init__(self, opt):
        """Constructor"""
        # Loading options
        self.shuffle = opt['train']['epoch_shuffle']
        self.root = opt['data']['data_path']

        # STFT paremeters
        self.segment_size = opt['data'].get('segment_size', 16000)
        self.rate = opt['data'].get('rate', 22050)
        self.n_fft = opt['data'].get('n_fft', 1024)
        self.hop = opt['data'].get('hop', 256)
        self.win_len = opt['data'].get('win_len', 1024)
        self.mel_fmin = opt['data'].get('mel_fmin', 0.0)
        self.mel_fmax = opt['data'].get('mel_fmax', 8000.0)

        self.stft = TacotronSTFT(filter_length=self.n_fft,
                                hop_length=self.hop,
                                win_length=self.win_len,
                                sampling_rate=self.rate,
                                mel_fmin=self.mel_fmin, mel_fmax=self.mel_fmax)

        self.df = pd.read_json(os.path.join(os.path.join(self.root, opt['data']['meta_file'])), orient='index')
        self.df['file'] = self.df.index
        self.df.reset_index(drop=True, inplace=True)

        # Filter instrument families
        self.inst_list = opt['data'].get('inst', None)
        if self.inst_list is not None:
            self.df = self.df[self.df['instrument_family_str'].isin(self.inst_list)]
        else:
            self.inst_list = list(self.df['instrument_family_str'].unique())
        print('Instruments families selected: {}'.format(self.inst_list))

        # filter pitch < 84 and pitch  > 24 
        self.df = self.df[(self.df['pitch'] < 84) & ( self.df['pitch'] > 24)]

    def __len__(self):
        return len(self.df)

    #def window_sample(self, samples, window_size):
    #    if samples.shape[0] > window_size:
    #       diff = samples.shape[0] - window_size
    #        idx = random.randint(0, diff)
    #        samples = samples[idx:idx + window_size]
    #    return samples

    def window_sample(self, samples, window_size):
        if MAX_INDEX_WAV > window_size:
            diff = MAX_INDEX_WAV - window_size
            idx = random.randint(0, diff)
            samples = samples[idx:idx + window_size]
        return samples

    def get_mel(self, audio_norm):
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec
    
    def normalize_audio(self, audio):
        # Already normalized
        if audio.min() >= -1 and audio.max() <= 1:
            return audio
        
        # Normalizing by the max/min value
        if np.abs(audio.min()) < np.abs(audio.max()):
            return audio / np.abs(audio.max())
        else:
            return audio / np.abs(audio.min())

    def arrange_data(self, path):
        #_, audio = scipy.io.wavfile.read(path)
        audio, sr = librosa.load(path, sr= self.rate, mono = True)
        audio = self.normalize_audio(audio)
        #audio = audio/ MAX_WAV_VALUE
        audio = audio.astype(np.float)
        audio = torch.from_numpy(audio).float()
        audio = self.window_sample(audio, self.segment_size)

        mel = self.get_mel(audio)

        return mel, audio
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dictionary
        """
        row = self.df.iloc[index]
        path =  os.path.join(self.root, 'audio/', row['file'] + '.wav')

        mel, audio = self.arrange_data(path)
        return {'mel': mel, 'audio': audio}