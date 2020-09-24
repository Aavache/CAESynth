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
from . import networks, loss
from . import network_utils as net_utils

class AEPClassModel(BaseModel):
    """
    """
    def __init__(self, opt, is_train= True):
        """Initialize a Autoencoder, which encoder also classifies pitch

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
            self.loss_names = ['recon', 'class']
            self.lambda_recon = opt['train']['lambda_recon']
            self.lambda_class = opt['train']['lambda_class']

            self.pitch_size = opt['data']['pitch_size']

            # define loss functions
            self.criterion_recon = loss.Distance(opt['train']['recon_mode'])
            self.criterion_class = nn.CrossEntropyLoss()

            # initialize optimizers
            self.optimizer_E = torch.optim.Adam(self.AE.enc_net_params, lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))
            self.optimizer_AE = torch.optim.Adam(self.AE.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))

            if opt['train'].get('load', False):
                self.load_networks(opt['train'].get('load_suffix', 'latest'))
                print('Network Loaded!')
    
    def set_input(self, data):
        self.data = data['data'].to(self.device)
        self.pitch_class = data['pitch'].to(self.device)

    def forward(self):
        """Run forward pass"""
        self.recon, latent = self.AE(self.data)
        split_size = latent.size(1) - self.pitch_size
        self.pitch_pred = torch.split(latent, split_size, dim=1)[1]
        self.pitch_pred = self.pitch_pred.squeeze(2).squeeze(2)

    def generate(self, data, trg_pitch):
        with torch.no_grad():
            return self.AE(data)
    
    def backward_AE(self):
        """Calculate GAN loss for discriminator D"""
        # Reconstruction loss already computed for Encoder
        self.loss_recon = self.criterion_recon(self.recon, self.data)* self.lambda_recon
        self.loss_recon.backward(retain_graph=True)

    def backward_E(self):
        """Calculate GAN loss for discriminator D"""
        # Reconstruction loss already computed for Encoder
        target_idx = torch.argmax(self.pitch_class, dim=1)
        self.loss_class = self.criterion_class(self.pitch_pred, target_idx)* self.lambda_class
        self.loss_class.backward(retain_graph=True)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward() # compute fake images and reconstruction images.

        #net_utils.set_requires_grad([self.Enc, self.Dec], True)
        self.optimizer_AE.zero_grad()   # set S gradients to zero
        self.backward_AE()      # calculate gradients for S
        self.optimizer_AE.step()  # update S's weights

        #net_utils.set_requires_grad([self.Enc], False)
        self.optimizer_E.zero_grad()  # set Decoder's gradients to zero
        self.backward_E()             # calculate gradients for Decoder
        self.optimizer_E.step()       # update G weights