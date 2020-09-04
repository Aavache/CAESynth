# External libs
import os
import torch
import numpy as np
import itertools
from collections import OrderedDict
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
# Internal libs
from .base_model import BaseModel
from . import networks, loss, util, signal_utils
from . import network_utils as net_utils

class Mag2PhaseModel(BaseModel):
    """
    """
    def __init__(self, opt, is_train = True):
        """Initialize Magnitude to Phase Model.

        Parameters:
            opt (dict)      - stores all the experiment flags; needs to be a subclass of BaseOptions
            is_train (bool) - Stage flag; {True: Training, False: Testing}
        """
        BaseModel.__init__(self, opt, is_train= True)

        assert opt['train']['batch_size'] == 1 

        # Instantiating networks
        self.mag2phase = networks.instantiate_mag2phase(opt)
        self.mag2phase.to(self.device)
        self.model_names = ['mag2phase']

        if is_train:  # define discriminators
            # specify the training losses you want to print out.
            self.lambda_recon = opt['train'].get('lambda_recon', 0.0)
            self.lambda_gd = opt['train'].get('lambda_gd', 0.0)
            self.lambda_if = opt['train'].get('lambda_if', 0.0)

            self.loss_names = []
            if self.lambda_recon != 0.0:
                self.loss_names.append('recon')
            if self.lambda_gd != 0.0:
                self.loss_names.append('gd')
            if self.lambda_if != 0.0:
                self.loss_names.append('if')
            assert len(self.loss_names) != 0

            # define loss functions
            self.criterion_recon = loss.Distance(opt['train']['recon_mode'])

            # initialize optimizers
            self.optimizer_m2p = torch.optim.Adam(self.mag2phase.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))

            if opt['train'].get('load', False):
                self.load_networks(opt['train'].get('load_suffix', 'latest'))
                print('Network Loaded!')
    
    def set_input(self, data):
        # Reference sample for both pitch and timbre
        data = data['data'].to(self.device)
        data = torch.transpose(data,0,3)
        data = data.squeeze(3)
        self.mag = data[:,0,:]
        self.phase = data[:,1,:]
        #self.phase = (data[:,1,:] + 2 * np.pi) % (2 * np.pi)

    def forward(self):
        """Run forward pass"""
        self.recon_phase = self.mag2phase(self.mag)

    def backward_m2p(self):
        # Reconstruction loss
        if self.lambda_recon != 0.0:
            self.loss_recon = self.criterion_recon(self.phase, self.recon_phase)* self.lambda_recon
        else:
            self.loss_recon = 0.0

        # Group Delay loss
        if self.lambda_gd != 0.0:
            recon_gd = signal_utils.instantaneous_frequency(self.recon_phase,0) # Doesnt work well
            #recon_gd = signal_utils.diff(self.recon_phase,-1) # Doesnt work well
            target_gd = signal_utils.instantaneous_frequency(self.phase, 0) 
            #target_gd = signal_utils.diff(self.phase,-1)
            self.loss_gd = self.criterion_recon(target_gd, recon_gd)* self.lambda_gd
        else:
            self.loss_gd = 0.0

        # Instantaneous Freq. Loss
        if self.lambda_if != 0.0:
            #recon_if = signal_utils.diff(self.recon_phase,0) # Freq axis finite difference, Doesnt work well
            #recon_if = signal_utils.instantaneous_frequency(self.recon_phase, 0) # Freq axis finite difference
            #target_if = signal_utils.diff(self.phase, 0)
            #target_if = signal_utils.instantaneous_frequency(self.phase, 0)
            #self.loss_if = self.criterion_recon(target_if - recon_if)* self.lambda_if
            self.loss_if = self.criterion_recon(self.phase, self.recon_phase)* self.lambda_if
        else:
            self.recon_if = 0.0

        total_loss = self.loss_recon + self.loss_gd + self.loss_if
        total_loss.backward(retain_graph=True)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()

        self.optimizer_m2p.zero_grad() 
        self.backward_m2p()
        self.optimizer_m2p.step()