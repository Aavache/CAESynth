# External libs
import os
import torch
import itertools
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
# Internal libs
from .base_model import BaseModel
from . import networks
from . import network_utils as net_utils

class AEModel(BaseModel):
    """
    """
    def __init__(self, opt, is_train= True):
        """Initialize Autoencoder.

        Parameters:
            opt (dict)      - stores all the experiment flags; needs to be a subclass of BaseOptions
            is_train (bool) - Stage flag; {True: Training, False: Testing}
        """
        BaseModel.__init__(self, opt, is_train= True)

        # Instantiating networks
        self.AE = networks.instantiate_autoencoder(opt)
        self.AE.to(self.device)
        self.model_names = ['AE']

        if is_train:  # define discriminators
            # Specify the training losses you want to print out.
            self.loss_names = ['recon']
            self.lambda_recon = opt['train']['lambda_recon']

            # define loss functions
            if opt['train']['recon_mode'] == 'l1':
                self.criterion_recon = nn.L1Loss()
            else:
                self.criterion_recon = nn.MSELoss()

            # initialize optimizers
            self.optimizer_AE = torch.optim.Adam(self.AE.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))

            if opt['train'].get('load', False):
                self.load_networks(opt['train'].get('load_suffix', 'latest'))
                print('Network Loaded!')
    
    def set_input(self, data):
        self.x = data['src_data'].to(self.device)
        self.y = data['trg_data'].to(self.device)

    def forward(self):
        """Run forward pass"""
        self.recon_src,  self.z_src = self.AE(self.x)
        self.recon_trg,  self.z_trg = self.AE(self.y) 

    def generate(self, data):
        with torch.no_grad():
            recon,  z = self.AE(data)
            return recon, z
    
    def backward_AE(self):
        # Reconstruction loss
        self.loss_recon = self.criterion_recon(self.recon_src, self.x)* self.lambda_recon
        self.loss_recon += self.criterion_recon(self.recon_trg, self.y)* self.lambda_recon

        self.loss_recon.backward(retain_graph=True)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()

        self.optimizer_AE.zero_grad() 
        self.backward_AE()  
        self.optimizer_AE.step() 