# This code implementation was mainly taken from: https://github.com/ss12f32v/GANsynth-pytorch
# External Libraries
import numpy as np
import torch 

class DataNormalizer(object):
    '''
    Data normalization class, before jumping on the implementation, you may have to compute the dataset statistics to 
    perform the normalization 
    '''
    def __init__(self, dataloader):
        self.dataloader = dataloader
        print('WARNING. Normalization parameters are hardcoded!')
        self.s_a = 0.0612 
        self.s_b = 0.0449
        if self.dataloader.mag_format == 'log':
            self.s_a = 0.0795
            self.s_b = 0.2990
        elif self.dataloader.mag_format == 'mel':
            # 16kH 256 mels librosa
            self.s_a = 0.0665
            self.s_b = 0.1189
        else:
            raise NotImplementedError
        # self._range_normalizer(magnitude_margin=0.8)
        print("s_a:", self.s_a)
        print("s_b:", self.s_b)

    def _range_normalizer(self, magnitude_margin):
        min_spec = 10000
        max_spec = -10000
        # Finding the min and max values in the trainset
        for batch_idx, data in enumerate(self.dataloader):
            spec = data['data'][0,:,:]
            if spec.min() < min_spec: min_spec=spec.min()
            if spec.max() > max_spec: max_spec=spec.max()
            if batch_idx % 1000 == 0:
                print("Iter: {}, Min Mag: {}, Max Mag: {}:".format(batch_idx, \
                             min_spec, max_spec))
        print("Done! >> Min Mag: {}, Max Mag: {}".format(min_spec, max_spec))
        self.s_a = magnitude_margin * (2.0 / (max_spec - min_spec))
        self.s_b = magnitude_margin * (-2.0 * min_spec / (max_spec - min_spec) - 1.0)
        
    def normalize(self, feature_map):
        a = np.asarray([self.s_a])[:, None, None]
        b = np.asarray([self.s_b])[:, None, None]
        a = torch.FloatTensor(a)
        b = torch.FloatTensor(b)
        feature_map = feature_map *a + b

        return feature_map
    
    def denormalize(self, spec, s_a=None, s_b=None):
        if s_a is None or s_b is None:
            s_a, s_b = self.s_a, self.s_b
        spec = (spec - s_b) / s_a
        return spec