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

class CAEModel(BaseModel):
    """
    """
    def __init__(self, opt, is_train= True):
        """Initialize the Pitch Condidtional Autoencoder.

        Parameters:
            opt (dict)      - stores all the experiment flags; needs to be a subclass of BaseOptions
            is_train (bool) - Stage flag; {True: Training, False: Testing}
        """
        BaseModel.__init__(self, opt, is_train= True)

        # Instantiating networks
        self.Enc = networks.instantiate_encoder(opt)
        self.Enc.to(self.device)
        self.model_names = ['Enc']

        self.Dec = networks.instantiate_decoder(opt)
        self.Dec.to(self.device)
        self.model_names += ['Dec']

        if is_train:  # define discriminators
            # Specify the training losses you want to print out.
            self.loss_names = ['recon']
            self.lambda_recon = opt['train']['lambda_recon']
            self.lambda_latent = opt['train']['lambda_latent']
            if self.lambda_latent > 0:
                self.loss_names += ['latent']

            # define loss functions
            if opt['train']['recon_mode'] == 'l1':
                self.criterion_recon = nn.L1Loss()
                self.criterion_latent = nn.L1Loss()
            else:
                self.criterion_recon = nn.MSELoss()
                self.criterion_latent = nn.MSELoss()

            # initialize optimizers
            self.optimizer_E = torch.optim.Adam(self.Enc.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))
            self.optimizer_D = torch.optim.Adam(self.Dec.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))

            if opt['train'].get('load', False):
                self.load_networks(opt['train'].get('load_suffix', 'latest'))
                print('Network Loaded!')
    
    def set_input(self, data):
        self.x = data['src_data']#.to(self.device)
        self.x_pitch = data['src_pitch']#.to(self.device)
        self.y = data['trg_data']#.to(self.device)
        self.y_pitch = data['trg_pitch']#.to(self.device)

    def forward(self):
        """Run forward pass"""
        self.zx_timbre = self.Enc(self.x) # Encoding Latent timbre of x sample
        self.xy_recon = self.Dec(self.zx_timbre, self.y_pitch) # Decode zx with y pitch
        self.zy_timbre = self.Enc(self.y) # Encoding Latent timbre of y sample
        self.yx_recon = self.Dec(self.zy_timbre, self.x_pitch) # Decode zy with x pitch

    def generate(self, data, trg_pitch):
        with torch.no_grad():
            z_timbre = self.Enc(data)
            recon = self.Dec(z_timbre, trg_pitch) # Decode zx with y pitch
            return recon, z_timbre
    
    def backward_D(self):
        """Calculate GAN loss for discriminator D"""
        # Reconstruction loss already computed for Encoder
        self.loss_recon.backward()

    def backward_E(self):
        """Calculate the loss for generators G"""
        # Reconstruction loss
        if self.lambda_recon > 0:
            self.loss_recon = self.criterion_recon(self.xy_recon, self.y)* self.lambda_recon
            self.loss_recon += self.criterion_recon(self.yx_recon, self.x)* self.lambda_recon
        else:
            self.loss_recon = 0

        if self.lambda_latent > 0:
            self.loss_latent = self.criterion_latent(self.zx_timbre, self.zy_timbre)* self.lambda_latent
        else:
            self.loss_latent = 0

        # Combined loss and calculate gradients
        loss_enc_total =  self.loss_recon + self.loss_latent
        loss_enc_total.backward(retain_graph=True)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.

        #net_utils.set_requires_grad([self.Enc, self.Dec], True)
        self.optimizer_E.zero_grad()   # set S gradients to zero
        self.backward_E()      # calculate gradients for S
        self.optimizer_E.step()  # update S's weights

        #net_utils.set_requires_grad([self.Enc], False)  # Gs require no gradients when optimizing Decoder
        self.optimizer_D.zero_grad()  # set Decoder's gradients to zero
        self.backward_D()             # calculate gradients for Decoder
        self.optimizer_D.step()       # update G weights