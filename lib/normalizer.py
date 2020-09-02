# This code implementation was mainly taken from: https://github.com/ss12f32v/GANsynth-pytorch
# External Libraries
import numpy as np
import torch 

class DataNormalizer(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        print('WARNING. Normalization parameters are hardcoded!')
        # When wav is normalized [-1,1]
        self.s_a = 0.0612 
        self.s_b = 0.0449
        if not self.dataloader.apply_if_mel:
            self.p_a = 1
            self.p_b = 0
        else:
            self.p_a = 0.0023
            self.p_b = -0.0067 

        # When wav is NOT normalized
        #self.p_a = 0.0026
        #self.p_b = 0.0185
        #self.s_a = 0.0341
        #self.s_b = -0.3292 
        
        #self.p_a = 0.0034997 # to erase
        #self.p_b = -0.010897 # to erase
        #self.s_a = 0.060437 # to erase
        #self.s_b = 0.034964 # to erase
        if self.dataloader.include_phase:
            #self._range_normalizer_with_IF(magnitude_margin=0.8, IF_margin=1.0)
            print("p_a:", self.p_a)
            print("p_b:", self.p_b)
        #else:
           #self._range_normalizer(magnitude_margin=0.8) 
        print("s_a:", self.s_a)
        print("s_b:", self.s_b)

    def _range_normalizer(self, magnitude_margin):
        min_spec = 10000
        max_spec = -10000

        for batch_idx, data in enumerate(self.dataloader):
            # training mel
            spec = data['src_data'][0,:,:]
            if spec.min() < min_spec: min_spec=spec.min()
            if spec.max() > max_spec: max_spec=spec.max()
            #if batch_idx > 100:
            #    break
        self.s_a = magnitude_margin * (2.0 / (max_spec - min_spec))
        self.s_b = magnitude_margin * (-2.0 * min_spec / (max_spec - min_spec) - 1.0)

    def _range_normalizer_with_IF(self, magnitude_margin, IF_margin):
        min_spec = 10000
        max_spec = -10000
        min_IF = 10000
        max_IF = -10000

        for batch_idx, data in enumerate(self.dataloader):
            # training mel
            spec = data['data'][0,:,:]
            IF = data['data'][1,:,:]
            if spec.min() < min_spec: min_spec=spec.min()
            if spec.max() > max_spec: max_spec=spec.max()

            if IF.min() < min_IF: min_IF=IF.min()
            if IF.max() > max_IF: max_IF=IF.max()
            if batch_idx % 1000 == 0:
                print(batch_idx)
            #if batch_idx > 100:
            #    break
        print(min_spec)
        print(max_spec)
        print(min_IF)
        print(max_IF)
        self.s_a = magnitude_margin * (2.0 / (max_spec - min_spec))
        self.s_b = magnitude_margin * (-2.0 * min_spec / (max_spec - min_spec) - 1.0)
        
        self.p_a = IF_margin * (2.0 / (max_IF - min_IF))
        self.p_b = IF_margin * (-2.0 * min_IF / (max_IF - min_IF) - 1.0)

    def normalize(self, feature_map):
        if self.dataloader.include_phase:
            a = np.asarray([self.s_a, self.p_a])[:, None, None]#[None, :, None, None]
            b = np.asarray([self.s_b, self.p_b])[:, None, None]#[None, :, None, None]
        else:
            a = np.asarray([self.s_a])[:, None, None]#[None, :, None, None]
            b = np.asarray([self.s_b])[:, None, None]#[None, :, None, None]            
        a = torch.FloatTensor(a)#.to(self.device)
        b = torch.FloatTensor(b)#.to(self.device)
        feature_map = feature_map *a + b

        return feature_map
    
    def denormalize(self, spec, s_a=None, s_b=None):
        if s_a is None or s_b is None:
            s_a, s_b = self.s_a, self.s_b
        spec = (spec - s_b) / s_a
        return spec

    def denormalize_IF(self, IF, p_a=None, p_b=None):
        if p_a is None or p_b is None:
            p_a, p_b = self.p_a, self.p_b
        IF = (IF - p_b) / p_a
        return IF