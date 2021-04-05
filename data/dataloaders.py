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

def expand(mat, desired_size=64):
    # while not math.log(mat.shape[1], 2).is_integer():
    while mat.shape[1] < desired_size:
        expand_vec = np.expand_dims(mat[:,-1],axis=1)
        mat = np.hstack((mat,expand_vec))
    return mat

def to_one_hot(class_size, index):
    return torch.Tensor(np.eye(class_size)[index])

class NSynthFSD(data.Dataset):

    """Pytorch dataset for Nsynth and FSD Wav dataset
    args:
        root: root dir containing examples.json and audio directory with
            wav files.
        is_train[bool]: indicates whether it is a training loading stage(True) or testing(False) 
    """

    def __init__(self, opt, is_train=True):
        """Constructor"""
        # Reading arguments
        self.is_train = is_train
        self.root = opt['data_path'] # ./data/
        self.nsynth_folder = opt['nsynth_folder'] 
        self.fsd_folder = opt['fsd_folder'] 

        if is_train:
            self.nsynth_meta_file = opt['nsynth_meta_file'] # train_family_source.json
        else: # Validation
            self.nsynth_meta_file = opt['nsynth_val_meta_file'] # val_family_source.json

        # Loading options
        self.shuffle = opt['epoch_shuffle']
        self.include_phase = opt['include_phase']
        self.segment_size = opt['segment_size']
        self.sample_size = opt.get('sample_size', 1)

        # STFT paremeters
        # if self.include_phase:
        self.phase_format = opt.get('phase_format', 'if')
        self.mag_format = opt.get('mag_format', 'mel')
        self.n_fft = opt.get('n_fft', 2048)
        self.hop = opt.get('hop', 256)
        self.win_len = opt.get('win_len', 2048)
        self.window = opt.get('window', 'hann')
        self.rate = opt.get('rate', 16000)
        self.n_mel = opt.get('n_mel', 256)
        self.fmin = opt.get('fmin', 27)
        self.fmax = opt.get('fmax', 11000)
        self.max_index_wav = int(0.9*self.rate)

        # Nysnth dataset
        self.nsynth_df = pd.read_json(os.path.join(os.path.join(self.root,self.nsynth_folder, self.nsynth_meta_file)), orient='index')
        self.nsynth_df['file'] = self.nsynth_df.index
        self.nsynth_df.reset_index(drop=True, inplace=True)

        # Filter pitch < 84 and pitch  > 24 
        self.pitch_range = opt.get('pitch_range', [24,84])
        self.pitch_class_size = self.pitch_range[1]
        self.nsynth_df = self.nsynth_df[(self.nsynth_df['pitch'] < self.pitch_range[1]) & (self.nsynth_df['pitch'] > self.pitch_range[0])]

        # FSD dataset
        self.fsd_df = pd.read_csv(os.path.join(self.root, self.fsd_folder, opt['fsd_meta_file']))

        # List of available instruments
        self.fsd_classes = sorted(list(self.fsd_df['labels'].unique()))
        self.nsynth_class_size = len(self.nsynth_df['family_source_id'].unique())

        self.timbre_class_size = len(self.fsd_classes) + self.nsynth_class_size
        
        # if is_train:
            # self.fsd_df = self.fsd_df[self.fsd_df['stage']=='train']
        # else:
            # self.fsd_df = self.fsd_df[self.fsd_df['stage']=='test']

        # Initializing Data Normalizer
        self.data_norm = DataNormalizer(self)
      
    def __len__(self):
        return len(self.nsynth_df) # There are many more samples in NSynth than FSD.

    def add_noise(self, samples):
        return samples * np.random.uniform(0.98, 1.02, len(samples)) + np.random.uniform(-0.005, 0.005, len(samples))

    def time_shift(self, samples):
        start = int(np.random.uniform(-4800,4800))
        if start >= 0:
            return np.r_[samples[start:], np.random.uniform(-0.001, 0.001, start)]
        else:
            return np.r_[np.random.uniform(-0.001,0.001, -start), samples[:start]]

    # TODO: move this methods to a utils script
    def pitch_shift(self, samples):
        steps = int(np.random.randint(-10, 10))
        return librosa.effects.pitch_shift(samples, self.rate, n_steps=steps)

    def augmend_data(self, samples):
        # if np.random.uniform(0,1) > 0.5:
            # samples = self.add_noise(samples)
        if np.random.uniform(0,1) > 0.5:
            samples = self.time_shift(samples)
        if np.random.uniform(0,1) > 0.5:
           samples = self.pitch_shift(samples)
        return samples

    def window_sample(self, samples, start, end):
        start = int(start * self.rate)
        end = samples.shape[0] if end==-1 else int(end*self.rate)
        samples = samples[start:end]
        if samples.shape[0] > self.segment_size:
            diff = samples.shape[0] - self.segment_size
            idx = random.randint(0, diff)
            samples = samples[idx:idx + self.segment_size]
        return samples
    
    def arrange_feature(self, path, start=0, end=-1, augment_enable=False):
        # ~7 times slower than scipy however is required to downsample the sampling rate
        # sample, sr = librosa.load(path, sr=self.rate, mono = True)
        sr, sample = scipy.io.wavfile.read(path)
        assert sr == self.rate
        if sample.max() > 1.5 and sample.min()<-1.5:
            sample = sample/ np.iinfo(np.int16).max

        sample = sample.astype(np.float)
        sample = self.window_sample(sample, start, end)

        # Data augmentation
        if augment_enable:
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
        spec = librosa.stft(sample, n_fft=self.n_fft, hop_length = self.hop, 
            win_length=self.win_len, window=self.window)
        mag = np.log(np.abs(spec) + 1.0e-6)[:self.n_fft//2]
        mag = expand(mag)
        if self.mag_format == 'mel':
            mag, _ = sign_op.specgrams_to_melspecgrams(magnitude = mag, mel_downscale=self.n_fft//(2*self.n_mel))
            #mag = librosa.feature.melspectrogram(S=mag**2, sr=self.rate, n_fft=self.n_fft, 

        # if self.include_phase:
        #     angle = np.angle(spec)
        #     if self.phase_format == 'phase':
        #         angle = expand(angle)[:self.n_fft//2]                 
        #         return mag, angle
        #     elif self.phase_format == 'unwrap':
        #         unwrapped_phase = sign_op.unwrap(angle)
        #         unwrapped_phase = expand(unwrapped_phase)[:self.n_fft//2]                
        #         return mag, unwrapped_phase
        #     elif self.phase_format == 'if':
        #         IF = sign_op.instantaneous_frequency(angle, time_axis=1)[:self.n_fft//2]
        #         IF = expand(IF)
        #         return mag, IF
        #     else:
        #         raise NotImplementedError
        return mag
    
    def __getitem__(self, nsynth_index):
        """
        Args:
            index (int): Index
        Returns:
            dictionary
        """
        if self.shuffle and nsynth_index == 0: # We shuffle the dataset metadata at the first iteration of every epoch
            self.nsynth_df = self.nsynth_df.sample(frac=1).reset_index(drop=True)
            self.fsd_df = self.fsd_df.sample(frac=1).reset_index(drop=True)
        
        # Nsynth sample
        nsynth_row = self.nsynth_df.iloc[nsynth_index]
        folder = 'nsynth-{}'.format(nsynth_row['stage'])
        name =  os.path.join(self.root,self.nsynth_folder, folder, 'audio/', nsynth_row['file'] + '.wav')
        nsynth_data = self.arrange_feature(name, start=0, end=1, augment_enable=False)
        pitch = to_one_hot(self.pitch_class_size, int(nsynth_row['pitch']))
        instr = to_one_hot(self.timbre_class_size, int(nsynth_row['family_source_id']))

        # FSD sample
        fsd_index =  nsynth_index % len(self.fsd_df)
        fsd_row = self.fsd_df.iloc[fsd_index]

        folder = fsd_row['stage']
        start = fsd_row['start']
        end = fsd_row['end']
        name =  os.path.join(self.root, self.fsd_folder, folder, str(fsd_row['fname']) + '.wav')

        fsd_data = self.arrange_feature(name, start=start, end=end, augment_enable=True)
        fsd_instr_idx = self.fsd_classes.index(fsd_row['labels'])
        fsd_pitch = to_one_hot(self.pitch_class_size, 0)
        fsd_instr = to_one_hot(self.timbre_class_size, self.nsynth_class_size + fsd_instr_idx)

        return {'nsynth_data': nsynth_data, 'nsynth_pitch': pitch, 'nsynth_instr': instr, 
                'fsd_data': fsd_data, 'fsd_pitch':fsd_pitch, 'fsd_instr':fsd_instr}

class NSynth(data.Dataset):

    """Pytorch dataset for NSynth wav dataset
    args:
        root: root dir containing examples.json and audio directory with
            wav files.
        is_train[bool]: indicates whether it is a training loading stage(True) or testing(False) 
    """

    def __init__(self, opt, is_train=True):
        """Constructor"""
        # Reading arguments
        self.is_train = is_train

        # Loading options
        self.shuffle = opt['epoch_shuffle']
        self.root = opt['data_path']
        if is_train:
            self.meta_file = opt['meta_file']
        else: # Validation
            self.meta_file = opt['val_meta_file']
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

    # def window_sample(self, samples):
    #     if self.max_index_wav < samples.shape[0]:
    #         samples = samples[:self.max_index_wav]
    #     if samples.shape[0] > self.segment_size:
    #         diff = samples.shape[0] - self.segment_size
    #         idx = random.randint(0, diff)
    #         samples = samples[idx:idx + self.segment_size]
    #     return samples

    def window_sample(self, samples, start, end):
        start = int(start * self.rate)
        end = samples.shape[0] if end==-1 else int(end*self.rate)
        samples = samples[start:end]
        if samples.shape[0] > self.segment_size:
            diff = samples.shape[0] - self.segment_size
            idx = random.randint(0, diff)
            samples = samples[idx:idx + self.segment_size]
        return samples
    
    def arrange_feature(self, path, start=0, end=1):
        # ~7 times slower than scipy.io
        #sample, sr = librosa.load(path, sr= self.rate, mono = True) 
        sr, sample = scipy.io.wavfile.read(path)
        assert sr == self.rate
        sample = sample/ np.iinfo(np.int16).max
        sample = sample.astype(np.float)
        sample = self.window_sample(sample, start, end)

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
            mag = librosa.feature.melspectrogram(y=sample, sr=self.rate, n_fft=self.n_fft, hop_length=self.hop, n_mels=self.n_mel,
                                                    fmin=self.fmin, fmax=self.fmax)
            mag = np.log(mag + 1.0e-6)
        else:
            spec = librosa.stft(sample, n_fft=self.n_fft, hop_length = self.hop, 
                win_length=self.win_len, window=self.window)
            mag = np.log(np.abs(spec) + 1.0e-6)[:self.n_fft//2]

        mag = expand(mag)
        # if self.include_phase:
        #     angle = np.angle(spec)
        #     if self.phase_format == 'phase':
        #         angle = expand(angle)[:self.n_fft//2]                 
        #         return mag, angle
        #     elif self.phase_format == 'unwrap':
        #         unwrapped_phase = sign_op.unwrap(angle)
        #         unwrapped_phase = expand(unwrapped_phase)[:self.n_fft//2]                
        #         return mag, unwrapped_phase
        #     elif self.phase_format == 'if':
        #         IF = sign_op.instantaneous_frequency(angle, time_axis=1)[:self.n_fft//2]
        #         IF = expand(IF)
        #         return mag, IF
        #     else:
        #         raise NotImplementedError
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

        row = self.df.iloc[index]
        folder = 'nsynth-{}'.format(row['stage'])
        name =  os.path.join(self.root, folder, 'audio/', row['file'] + '.wav')
        data = self.arrange_feature(name)
        
        pitch = to_one_hot(self.pitch_class_size, int(row['pitch']))
        instr = to_one_hot(self.timbre_class_size, int(row['family_source_id']))

        return {'data': data, 'pitch': pitch, 'instr': instr}