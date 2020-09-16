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

    def __init__(self, opt):
        """Constructor"""
        # Loading options
        self.shuffle = opt['train']['epoch_shuffle']
        self.root = opt['data']['data_path']
        self.include_phase = True if opt['model']['in_ch'] == 2 else False 
        self.segment_size = opt['data']['segment_size']
        self.sample_size = opt['data'].get('sample_size', 2)
        self.mag_format = opt['data'].get('mag_format', 'mel')
        self.phase_format = opt['data'].get('phase_format', 'mel_if')
        self.if_atten = opt['data'].get('if_atten', None)

        # STFT paremeters
        self.n_fft = opt['data'].get('n_fft', 2048)
        self.hop = opt['data'].get('hop', 512)
        self.win_len = opt['data'].get('win_len', 2048)

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

        # Initializing Data Normalizer
        self.data_norm = DataNormalizer(self)
      
    def __len__(self):
        return len(self.df)

    def window_sample(self, samples, window_size):
        if samples.shape[0] > window_size:
            diff = samples.shape[0] - window_size
            idx = random.randint(0, diff)
            samples = samples[idx:idx + window_size]
        return samples
    
    def attenuate_phase(self, data):
        '''We attenuate the phase information by the 
        '''
        if self.if_atten is None:
            return data
        elif self.if_atten == 'fade':
            mag = data[0,:,:]
            fader = (mag - mag.min())/(mag.max() - mag.min()) # Normalize 0 to 1
            data[1,:,:] = data[1,:,:] * fader
        elif self.if_atten == 'mask':
            mag = data[0,:,:]
            fader = (mag - mag.min())/(mag.max() - mag.min()) # Normalize 0 to 1
            mask = torch.zeros_like(mag)
            mask[fader>0.5] = 1
            data[1,:,:] = data[1,:,:] * mask
        return data
    
    def arrange_feature(self, path, pitch):
        _, sample = scipy.io.wavfile.read(path)
        sample = sample/ np.iinfo(np.int16).max
        sample = sample.astype(np.float)
        sample = self.window_sample(sample, self.segment_size)

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

        # Fading Phase
        data = self.attenuate_phase(data)

        pitch = to_one_hot(128, int(pitch))#.to(self.device)
        return data, pitch

    def compute_features(self, sample):
        spec = librosa.stft(sample, n_fft=self.n_fft, \
                hop_length = self.hop, win_length=self.win_len)
        
        mag = np.log(np.abs(spec)+ 1.0e-6)[:self.n_fft//2]
        mag = expand(mag)
        if self.mag_format == 'mel':
            mag, _ = sign_op.specgrams_to_melspecgrams(magnitude = mag)

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
            elif self.phase_format == 'mel_if':
                IF = sign_op.instantaneous_frequency(angle, time_axis=1)[:self.n_fft//2]
                IF = expand(IF)
                _, mel_if = sign_op.specgrams_to_melspecgrams(IF=IF)
                return mag, mel_if
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

        anc_instr = torch.tensor(int(anc_row['instrument_family']))
        dip1_instr = torch.tensor(int(dip1_row['instrument_family']))
        dip2_instr = torch.tensor(int(dip2_row['instrument_family']))

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

        src_instr = torch.tensor(int(src_row['instrument_family']))
        trg_instr = torch.tensor(int(trg_row['instrument_family']))

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
        name =  os.path.join(self.root, 'audio/', row['file'] + '.wav')
        data, pitch = self.arrange_feature(name, row['pitch'])
        instr = torch.tensor(int(row['instrument_family']))

        return {'data': data, 'pitch': pitch, 'instr': instr}