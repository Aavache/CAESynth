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

def expand(mat):
    while not math.log(mat.shape[1], 2).is_integer():
        expand_vec = np.expand_dims(mat[:,-1],axis=1)
        mat = np.hstack((mat,expand_vec))
    return mat

def to_one_hot(class_size, index):
    return torch.Tensor(np.eye(class_size)[index])

class NSynth(data.Dataset):

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

    def __init__(self, opt, size=None, is_train=True):
        """Constructor"""
        # Reading arguments
        self.size=size
        self.is_train = is_train

        # Loading options
        self.shuffle = opt['epoch_shuffle']
        self.root = opt['data_path'] # ./data/nsynth/
        if is_train:
            self.meta_file = opt['meta_file'] # train_family_source.json
        else: # Validation
            self.meta_file = opt['val_meta_file'] # val_family_source.json
        self.include_phase = opt['include_phase']
        self.segment_size = opt['segment_size']
        self.sample_size = opt.get('sample_size', 2)
        self.mag_format = opt.get('mag_format', 'mel')
        self.phase_format = opt.get('phase_format', 'if')
        self.augmentation_enabled = opt.get('augmentation_enabled', True)

        # STFT paremeters
        self.n_fft = opt.get('n_fft', 2048)
        self.hop = opt.get('hop', 256)
        self.win_len = opt.get('win_len', 2048)
        self.window = opt.get('window', 'hann')
        self.rate = opt.get('rate', 16000)
        self.n_mel = opt.get('n_mel', 256)
        self.fmin = opt.get('fmin', 27)
        self.fmax = opt.get('fmax', 11000)
        self.max_index_wav = int(self.rate)

        self.mel_filter = librosa.filters.mel(sr=self.rate, n_fft=self.n_fft, 
                                       n_mels=self.n_mel, fmin=self.fmin, fmax=self.fmax)

        self.df = pd.read_json(os.path.join(os.path.join(self.root, self.meta_file)), orient='index')
        self.df['file'] = self.df.index
        self.df.reset_index(drop=True, inplace=True)

        # List of available instruments
        self.timbre_class_size = len(self.df['family_source_id'].unique())

        # Filter pitch < 84 and pitch  > 24 
        self.pitch_range = opt.get('pitch_range', [24,84])
        self.pitch_class_size = self.pitch_range[1]
        self.df = self.df[(self.df['pitch'] < self.pitch_range[1]) & ( self.df['pitch'] > self.pitch_range[0])]

        # Initializing Data Normalizer
        self.data_norm = DataNormalizer(self)
      
    def __len__(self):
        if self.size is not None:
            return self.size
        else:
            return len(self.df)

    def add_noise(self, samples):
        return samples * np.random.uniform(0.98, 1.02, len(samples)) + np.random.uniform(-0.005, 0.005, len(samples))

    def time_shift(self, samples):
        start = int(np.random.uniform(-4800,4800))
        if start >= 0:
            return np.r_[samples[start:], np.random.uniform(-0.001, 0.001, start)]
        else:
            return np.r_[np.random.uniform(-0.001,0.001, -start), samples[:start]]
    
    def augmend_data(self, samples):
        # if np.random.uniform(0,1) > 0.5:
            # samples = self.add_noise(samples)
        # if np.random.uniform(0,1) > 0.5:
            # samples = self.time_shift(samples)s
        # if np.random.uniform(0,1) > 0.5:
            # samples = self.pitch_shift(samples)
        return samples

    def window_sample(self, samples):
        if self.max_index_wav < samples.shape[0]:
            samples = samples[:self.max_index_wav]
        if samples.shape[0] > self.segment_size:
            diff = samples.shape[0] - self.segment_size
            idx = random.randint(0, diff)
            samples = samples[idx:idx + self.segment_size]
        return samples
    
    def arrange_feature(self, path):
        # ~7 times slower than scipy
        #sample, sr = librosa.load(path, sr= self.rate, mono = True) 
        sr, sample = scipy.io.wavfile.read(path)
        assert sr == self.rate
        sample = sample/ np.iinfo(np.int16).max
        sample = sample.astype(np.float)
        sample = self.window_sample(sample)

        # Data augmentation
        if self.augmentation_enabled:
            sample = self.augmend_data(sample)

        if self.include_phase:
            mag, phase = self.compute_features(sample)
            mag = torch.from_numpy(mag).float()
            mag = mag.unsqueeze(0)
            phase = torch.from_numpy(phase).float()
            phase = phase.unsqueeze(0)
            data = torch.cat([mag, phase], dim = 0)
        else:
            mag = self.compute_features(sample)
            mag = torch.from_numpy(mag).float()
            data = mag.unsqueeze(0)

        # Normalize features
        data = self.data_norm.normalize(data)#.to(self.device))
        return data

    def compute_features(self, sample):
        if self.mag_format == 'mel':
            # mag, _ = sign_op.specgrams_to_melspecgrams(magnitude = mag, mel_downscale=self.n_fft//(2*self.n_mel))
            mag = librosa.feature.melspectrogram(y=sample, sr=self.rate, n_fft=self.n_fft, hop_length=self.hop, n_mels=self.n_mel,
                                                    fmin=self.fmin, fmax=self.fmax)
            mag = np.log(mag + 1.0e-6)
        else:
            spec = librosa.stft(sample, n_fft=self.n_fft, hop_length = self.hop, 
                win_length=self.win_len, window=self.window)
            mag = np.log(np.abs(spec) + 1.0e-6)[:self.n_fft//2]

        mag = expand(mag)
        if self.include_phase:
            angle = np.angle(spec)
            if self.phase_format == 'phase':
                angle = expand(angle)[:self.n_fft//2]                 
                return mag, angle
            elif self.phase_format == 'unwrap':
                unwrapped_phase = sign_op.unwrap(angle)
                unwrapped_phase = expand(unwrapped_phase)[:self.n_fft//2]                
                return mag, unwrapped_phase
            elif self.phase_format == 'if':
                IF = sign_op.instantaneous_frequency(angle, time_axis=1)[:self.n_fft//2]
                IF = expand(IF)
                return mag, IF
            # elif self.phase_format == 'mel_if':
                # IF = sign_op.instantaneous_frequency(angle, time_axis=1)[:self.n_fft//2]
                # IF = expand(IF)
                # _, mel_if = sign_op.specgrams_to_melspecgrams(IF=IF)
                # return mag, mel_if
            else:
                raise NotImplementedError
        return mag
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dictionary
        """
        if self.shuffle and index == 0: # We shuffle the dataset metadata at the first iteration of every epoch
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        if self.sample_size == 3:
            return self.get_triplet_item(index)
        elif self.sample_size == 2:
            return self.get_doublet_item(index)
        else:
            return self.get_single_item(index)

    def get_triplet_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dictionary
        """
        anc_row = self.df.iloc[index]
        # Sampling first, dipole, positive in pitch and negative in timber
        dip1_row = self.df[(self.df['instrument'] != anc_row['instrument']) & 
                        (self.df['pitch'] == anc_row['pitch'])].sample(n=1)

        # Sampling second, dipole, positive in timbre and negative in pitch
        dip2_row = self.df[(self.df['instrument'] == anc_row['instrument']) & 
                        (self.df['pitch'] != anc_row['pitch'])].sample(n=1)

        anc_name =  os.path.join(self.root, 'audio/', anc_row['file'] + '.wav')
        dip1_name =  os.path.join(self.root, 'audio/', dip1_row.iloc[0]['file'] + '.wav')
        dip2_name =  os.path.join(self.root, 'audio/', dip2_row.iloc[0]['file'] + '.wav')

        anc_data, anc_pitch = self.arrange_feature(anc_name, anc_row['pitch'])
        dip1_data, dip1_pitch = self.arrange_feature(dip1_name, dip1_row['pitch'])
        dip2_data, dip2_pitch = self.arrange_feature(dip2_name, dip2_row['pitch'])

        # anc_instr = torch.tensor(self.instru_list.index(int(anc_row['instrument'])))
        # dip1_instr = torch.tensor(self.instru_list.index(int(dip1_row['instrument'])))
        # dip2_instr = torch.tensor(self.instru_list.index(int(dip2_row['instrument'])))

        anc_instr = to_one_hot(self.timbre_class_size, self.instru_list.index(int(anc_row['instrument'])))
        dip1_instr = to_one_hot(self.timbre_class_size, self.instru_list.index(int(dip1_row['instrument'])))
        dip2_instr = to_one_hot(self.timbre_class_size, self.instru_list.index(int(dip2_row['instrument'])))

        return {'anc_data': anc_data, 'anc_pitch': anc_pitch, 'anc_instr': anc_instr,
                'dip1_data': dip1_data, 'dip1_pitch': dip1_pitch, 'dip1_instr': dip1_instr,
                'dip2_data': dip2_data, 'dip2_pitch': dip2_pitch, 'dip2_instr': dip2_instr}

    def get_doublet_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dictionary
        """
        src_row = self.df.iloc[index]
        trg_row = self.df[(self.df['instrument'] == src_row['instrument']) & 
                        (self.df['velocity'] == src_row['velocity'])].sample(n=1)

        src_name =  os.path.join(self.root, 'audio/', src_row['file'] + '.wav')
        trg_name =  os.path.join(self.root, 'audio/', trg_row.iloc[0]['file'] + '.wav')

        src_data, src_pitch = self.arrange_feature(src_name, src_row['pitch'])
        trg_data, trg_pitch = self.arrange_feature(trg_name, trg_row['pitch'])

        # src_instr = torch.tensor(self.instru_list.index(int(src_row['instrument'])))
        # trg_instr = torch.tensor(self.instru_list.index(int(trg_row['instrument'])))

        src_instr = to_one_hot(self.timbre_class_size, self.instru_list.index(int(src_row['instrument'])))
        trg_instr = to_one_hot(self.timbre_class_size, self.instru_list.index(int(trg_row['instrument'])))

        return {'src_data': src_data, 'src_pitch': src_pitch, 'src_instr': src_instr,
                'trg_data': trg_data, 'trg_pitch': trg_pitch, 'trg_instr': trg_instr}

    def get_single_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dictionary
        """
        row = self.df.iloc[index]
        folder = 'nsynth-{}'.format(row['stage'])
        name =  os.path.join(self.root, folder, 'audio/', row['file'] + '.wav')
        data = self.arrange_feature(name)
        
        pitch = to_one_hot(self.pitch_class_size, int(row['pitch']))
        instr = to_one_hot(self.timbre_class_size, int(row['family_source_id']))

        return {'data': data, 'pitch': pitch, 'instr': instr}