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

class PhaseAEModel(BaseModel):
    """
    """
    def __init__(self, opt, is_train= True):
        """Initialize Phase Autoencoder.

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
            self.loss_names = ['recon']
            self.lambda_recon = opt['train']['lambda_recon']

            self.optimizer_AE = torch.optim.Adam(self.AE.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))
            # define loss functions
            if opt['train']['recon_mode'] == 'l1':
                self.criterion_recon = nn.L1Loss()
            else:
                self.criterion_recon = nn.MSELoss()

            if opt['train'].get('load', False):
                self.load_networks(opt['train'].get('load_suffix', 'latest'))
                print('Network Loaded!')
    
    def set_input(self, data):
        # Reference sample for both pitch and timbre
        data = data['data'].to(self.device)

        self.data = data[:,1:2,:,:] # Slicing the phase channel

    def forward(self):
        """Run forward pass"""
        self.recon, _ = self.AE(self.data)

    def backward_AE(self):
        self.loss_recon = self.criterion_recon(self.recon, self.data) * self.lambda_recon
        self.loss_recon.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()

        self.optimizer_AE.zero_grad() 
        self.backward_AE()  
        self.optimizer_AE.step() 