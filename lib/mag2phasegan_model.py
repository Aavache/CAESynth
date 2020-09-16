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

class Mag2PhaseGANModel(BaseModel):
    """
    """
    def __init__(self, opt, is_train = True):
        """Initialize Magnitude to Phase GAN Model.

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
            self.loss_names = ['g_gan', 'd_gan']

            self.DISC = networks.instantiate_m2p_disc(opt)
            self.DISC.to(self.device)
            self.model_names += ['DISC']

            # specify the training losses you want to print out.
            self.lambda_gan = opt['train']['lambda_gan']
            self.lambda_gp = opt['train'].get('lambda_gp', 0.0)
            self.lambda_recon = opt['train'].get('lambda_recon', 0.0)
            self.lambda_gd = opt['train'].get('lambda_gd', 0.0)
            self.lambda_if = opt['train'].get('lambda_if', 0.0)

            if self.lambda_gp != 0.0:
                self.loss_names.append('gp')
            if self.lambda_recon != 0.0:
                self.loss_names.append('recon')
            if self.lambda_gd != 0.0:
                self.loss_names.append('gd')
            if self.lambda_if != 0.0:
                self.loss_names.append('if')

            assert len(self.loss_names) != 0

            # define loss functions
            self.criterion_recon = loss.Distance(opt['train']['recon_mode'])
            self.criterion_gan = loss.GANLoss(opt['train']['gan_mode'], self.device)

            # initialize optimizers
            self.optimizer_m2p = torch.optim.Adam(self.mag2phase.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))
            self.optimizer_disc = torch.optim.Adam(self.DISC.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))

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
    
    def backward_disc_basic(self, DISC, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """

        # Real
        pred_real = DISC(real)
        loss_D_real = self.criterion_gan(pred_real, True)
        loss_D_real.backward()
        
        # Fake
        pred_fake = DISC(fake.detach())
        loss_D_fake = self.criterion_gan(pred_fake, False)
        loss_D_fake.backward()

        # Gradient Penalty
        if self.lambda_gp > 0.0:
            real_data = real.type(torch.FloatTensor).to(self.device).data
            fake_data = fake.type(torch.FloatTensor).to(self.device).data
            loss_gp = loss.cal_gradient_penalty(DISC, real_data, fake_data, self.device, 
                                                lambda_gp= self.lambda_gp)[0]
            loss_gp.backward()
        else:
            loss_gp = 0.0
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake)*0.5

        return loss_D, loss_gp

    def backward_disc(self):
        """Calculate GAN loss for discriminator D_A"""
        loss_d, loss_gp = self.backward_disc_basic(self.DISC, self.phase, self.recon_phase)

        self.loss_d_gan = loss_d 
        self.loss_gp = loss_gp

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

        self.loss_g_gan = self.criterion_gan(self.DISC(self.recon_phase), True)* self.lambda_gan 

        total_loss = self.loss_recon + self.loss_gd + self.loss_if + self.loss_g_gan
        total_loss.backward(retain_graph=True)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()

        self.optimizer_m2p.zero_grad() 
        self.backward_m2p()
        self.optimizer_m2p.step()

        self.optimizer_disc.zero_grad() 
        self.backward_disc()
        self.optimizer_disc.step()