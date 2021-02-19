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

class ProgDualAEModel(BaseModel):
    """
    Progressive growing Autoencoder model
    """
    def __init__(self, opt, is_train= True):
        """Initialize Dual Autoencoder Model.

        Parameters:
            opt (dict)      - stores all the experiment flags; needs to be a subclass of BaseOptions
            is_train (bool) - Stage flag; {True: Training, False: Testing}
        """
        BaseModel.__init__(self, opt, is_train= True)

        # Instantiating networks
        self.AE = networks.instantiate_net(opt['model']['ae_net'])
        self.AE.to(self.device)
        self.model_names = ['AE']

        self.max_phase =  len(self.AE.enc_t.enc_net)-1
        self.fadein_iters = opt['train']['fadein_iters']
        self.stable_iters = opt['train']['stable_iters']
        self.iters = 0

        self.net_status = "stable" # {stable: last trained block is stable , fadein: we are introducing a new block}
        self.net_alpha = 1.0
        if is_train:
            self.net_phase = 0 # the phase denotes the number of conv blocks from the latent space.
        else:
            self.net_phase = self.max_phase

        current_config = {'phase': self.net_phase, 'alpha': self.net_alpha, 'status':self.net_status}
        self.AE.update_config(current_config)

        if is_train:  # define discriminators
            # Specify the training losses you want to print out.
            self.loss_names = ['recon']
            self.lambda_recon = opt['train']['lambda_recon']

            # Define loss functions
            self.criterion_recon = loss.Distance(opt['train']['recon_mode'])

            self.optimizer_AE = torch.optim.Adam(self.AE.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))
            print('Learning Rate: {}'.format(opt['train']['lr']))
            if opt['train'].get('load', False):
                self.load_networks(opt['train'].get('load_suffix', 'latest'))
                print('Network Loaded!')
    
    def set_input(self, data):
        self.data = self.resize(data['data'].to(self.device))

    def forward(self):
        """Run forward pass"""
        self.recon = self.AE(self.data)[0]

    def validate(self, data):
        with torch.no_grad():
            self.forward()
            self.compute_losses()

    def compute_losses(self):
        # Reconstruction loss
        self.loss_recon = self.criterion_recon(self.recon, self.data)* self.lambda_recon

    def backward_AE(self):
        self.compute_losses()

        total_loss = self.loss_recon
        total_loss.backward(retain_graph=True)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()

        self.optimizer_AE.zero_grad() 
        self.backward_AE()  
        # torch.nn.utils.clip_grad_norm(self.AE.parameters(), 1)
        self.optimizer_AE.step()
        
        # Progressive Growing parameters update!
        self.update_net_status()
    
    def resize(self, data):
        current_phase = self.max_phase
        while current_phase > self.net_phase:
            if current_phase != self.max_phase:
                data = net_utils.down_sample2x2(data)
            current_phase -= 1
        return data.detach()
    
    def print_net_status(self):
        print("Net status has changed to: status: {}, phase: {}, alpha: {}".format(self.net_status, \
                        self.net_phase, self.net_alpha))

    def update_net_status(self):
        self.iters += 1
        has_status_changed = False
        if self.net_phase == self.max_phase:
            if self.net_status == "fadein":
                if self.iters % self.fadein_iters == 0:
                    self.net_status = "stable"
                    has_status_changed = True
        elif self.net_phase == 0:
            if self.iters % self.stable_iters == 0:
                self.net_status = "fadein"
                self.net_phase += 1
                has_status_changed = True
        else:
            if self.net_status == "stable":
                if self.iters % self.stable_iters == 0:
                    self.net_status = "fadein"
                    self.net_phase += 1
                    has_status_changed = True

            else: # "fadein" 
                if self.iters % self.fadein_iters == 0:
                    self.net_status = "stable"
                    has_status_changed = True
        
        if self.net_status == "fadein":
            fadein_perc = (self.iters % self.fadein_iters)/self.fadein_iters
            self.net_alpha = 1 - fadein_perc
        else:
            self.net_alpha = 1

        current_config = {'phase': self.net_phase, 'alpha': self.net_alpha, 'status':self.net_status}
        self.AE.update_config(current_config)

        if has_status_changed:
            self.print_net_status()
