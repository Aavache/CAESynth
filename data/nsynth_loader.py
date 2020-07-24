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
        self.shuffle = opt['train']['epoch_shuffle']
        self.root = opt['data']['data_path']
        self.include_phase = True if opt['model']['in_ch'] == 2 else False 
        self.segment_size = opt['data']['segment_size']
        #self.filenames = glob.glob(os.path.join(self.root, "audio/*.wav"))

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
    
    def arrange_feature(self, path, pitch):
        _, sample = scipy.io.wavfile.read(path)
        sample = sample.astype(np.float)
        sample = self.window_sample(sample, self.segment_size)

        if self.include_phase:
            logmel, mel_p = self.compute_features(sample)
            mel_p = torch.from_numpy(mel_p).float()
            mel_p = mel_p.unsqueeze(0)
            logmel = torch.from_numpy(logmel).float()
            logmel = logmel.unsqueeze(0)
            data = torch.cat([logmel, mel_p], dim = 0)
        else:
            logmel = self.compute_features(sample)
            logmel = torch.from_numpy(logmel).float()
            data = logmel.unsqueeze(0)

        # Normalize features
        data = self.data_norm.normalize(data)#.to(self.device))

        pitch = to_one_hot(128, int(pitch))#.to(self.device)
        return data, pitch

    def compute_features(self, sample):
        spec = librosa.stft(sample, n_fft=2048, hop_length = 512)
        
        magnitude = np.log(np.abs(spec)+ 1.0e-6)[:1024]
        magnitude = expand(magnitude)
        if self.include_phase:
            angle =np.angle(spec)
            IF = sign_op.instantaneous_frequency(angle, time_axis=1)[:1024]
            IF = expand(IF)
            logmel, mel_p = sign_op.specgrams_to_melspecgrams(magnitude, IF)
            return logmel, mel_p
        else:
            logmel = sign_op.specgrams_to_melspecgrams(magnitude)
            return logmel 

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dictionary
        """
        if self.shuffle and index == 0:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        src_row = self.df.iloc[index]
        trg_row = self.df[(self.df['instrument'] == src_row['instrument']) & 
                        (self.df['velocity'] == src_row['velocity'])].sample(n=1)

        src_name =  os.path.join(self.root, 'audio/', src_row['file'] + '.wav')
        trg_name =  os.path.join(self.root, 'audio/', trg_row.iloc[0]['file'] + '.wav')

        src_data, src_pitch = self.arrange_feature(src_name, src_row['pitch'])
        trg_data, trg_pitch = self.arrange_feature(trg_name, trg_row['pitch'])

        src_instr = torch.tensor(int(src_row['instrument_family']))
        trg_instr = torch.tensor(int(trg_row['instrument_family']))

        # return {'data': data, 'pitch': pitch}
        return {'src_data': src_data, 'src_pitch': src_pitch, 'src_instr': src_instr,
                'trg_data': trg_data, 'trg_pitch': trg_pitch, 'trg_instr': trg_instr}