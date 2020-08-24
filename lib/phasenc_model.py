# External libs
import os
import torch
import itertools
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
# Internal libs
from .base_model import BaseModel
from . import networks, loss
from . import network_utils as net_utils

class PhasEncModel(BaseModel):
    """
    """
    def __init__(self, opt, is_train= True):
        """Initialize Phase Autoencoder with Triplet Loss.

        Parameters:
            opt (dict)      - stores all the experiment flags; needs to be a subclass of BaseOptions
            is_train (bool) - Stage flag; {True: Training, False: Testing}
        """
        BaseModel.__init__(self, opt, is_train= True)

        opt['model']['in_ch'] = 1
        
        # Instantiating networks
        self.AE = networks.instantiate_autoencoder(opt)
        self.AE.to(self.device)
        self.model_names = ['AE']

        if is_train:  # define discriminators
            # Specify the training losses you want to print out.
            self.loss_names = ['pitch_trip', 'timbre_trip']
            self.lambda_pitch_trp = opt['train']['lambda_pitch_trp']
            self.lambda_timbre_trp = opt['train']['lambda_timbre_trp']

            self.criterion_triplet = loss.TripletLoss()
            self.optimizer_E = torch.optim.Adam(self.AE.enc_net_params, lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))

            if opt['train'].get('load', False):
                self.load_networks(opt['train'].get('load_suffix', 'latest'))
                print('Network Loaded!')
    
    def set_input(self, data):
        # Reference sample for both pitch and timbre
        self.anchor = data['anc_data'].to(self.device)
        self.anchor = self.anchor[:,1:2,:,:] # Slicing the phase channel
        # Positive pitch negative timbre sample
        self.dip1 = data['dip1_data'].to(self.device) 
        self.dip1 = self.dip1[:,1:2,:,:]
        # Positive timbre negative pitch sample
        self.dip2 = data['dip2_data'].to(self.device) 
        self.dip2 = self.dip2[:,1:2,:,:]

    def forward(self):
        """Run forward pass"""
        self.z_anc = self.AE.encode(self.anchor)
        self.z_dip1 = self.AE.encode(self.dip1)
        self.z_dip2 = self.AE.encode(self.dip2) 

    def backward_E(self):
        anc_tim, anc_pitch = torch.chunk(self.z_anc, chunks=2, dim=1)
        dip1_tim, dip1_pitch = torch.chunk(self.z_dip1, chunks=2, dim=1)
        dip2_tim, dip2_pitch = torch.chunk(self.z_dip2, chunks=2, dim=1)

        # Triple Pitch Loss
        self.loss_pitch_trip = self.criterion_triplet(anc_pitch, dip1_pitch, dip2_pitch)* self.lambda_pitch_trp
        # Triple Timbre Loss
        self.loss_timbre_trip = self.criterion_triplet(anc_tim, dip2_tim, dip1_tim)* self.lambda_timbre_trp

        total_loss = self.loss_timbre_trip + self.loss_pitch_trip
        total_loss.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()

        self.optimizer_E.zero_grad() 
        self.backward_E()  
        self.optimizer_E.step() 