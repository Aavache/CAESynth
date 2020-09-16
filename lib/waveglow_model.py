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
from . import networks, glow, loss
from . import network_utils as net_utils

class WaveGlowModel(BaseModel):
    """
    """
    def __init__(self, opt, is_train= True):
        """Initialize WaveGlow.

        Parameters:
            opt (dict)      - stores all the experiment flags; needs to be a subclass of BaseOptions
            is_train (bool) - Stage flag; {True: Training, False: Testing}
        """
        BaseModel.__init__(self, opt, is_train= True)

        # Instantiating networks
        self.vocoder = glow.WaveGlow(**opt['model']['waveglow_config']).to(self.device)
        self.model_names = ['vocoder']

        if is_train:  # define discriminators
            # Specify the training losses you want to print out.
            self.loss_names = ['voc']
            self.lambda_voc = opt['train'].get('lambda_voc', 1.0)

            # Define loss functions
            self.criterion = glow.WaveGlowLoss()

            # Initialize optimizers
            self.optimizer_voc = torch.optim.Adam(self.vocoder.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))

            if opt['train'].get('load', False):
                self.load_networks(opt['train'].get('load_suffix', 'latest'))
                print('Network Loaded!')
    
    def set_input(self, data):
        self.mel = torch.autograd.Variable(data['mel']).to(self.device)
        self.audio = data['audio'].to(self.device)

    def forward(self):
        """Run forward pass"""
        self.outputs = self.vocoder((self.mel, self.audio))

    def generate(self, data):
        raise NotImplementedError

    def backward_voc(self):
        ''''''
        self.loss_voc = self.criterion(self.outputs)* self.lambda_voc
        self.loss_voc.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()

        self.optimizer_voc.zero_grad() 
        self.backward_voc()  
        self.optimizer_voc.step()
