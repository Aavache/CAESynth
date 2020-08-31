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
from . import networks, loss, util
from . import network_utils as net_utils

class ProgAEModel(BaseModel):
    """
    """
    def __init__(self, opt, is_train= True):
        """Initialize PROGressive growing AutoEncoder.

        Parameters:
            opt (dict)      - stores all the experiment flags; needs to be a subclass of BaseOptions
            is_train (bool) - Stage flag; {True: Training, False: Testing}
        """
        BaseModel.__init__(self, opt, is_train= True)

        # Instantiating networks
        self.encoder = networks.instantiate_encoder(opt)
        self.encoder.to(self.device)
        self.model_names = ['encoder']

        self.decoder = networks.instantiate_decoder(opt)
        self.decoder.to(self.device)
        self.model_names += ['decoder']

        print('Model architecture: {}'.format(self.model_names))
        # Init network status
        self.max_phase =  len(self.encoder.enc_net)-1
        self.fadein_iters = opt['train']['fadein_iters']
        self.stable_iters = opt['train']['stable_iters']
        self.iters = 0

        self.net_phase = 0 # the phase denotes the number of conv blocks from the latent space.
        self.net_status = "stable" # {stable: last trained block is stable , fadein: we are introducing a new block}
        self.net_alpha = 1.0

        current_config = {'phase': self.net_phase, 'alpha': self.net_alpha, 'status':self.net_status}
        self.encoder.config = current_config
        self.decoder.config = current_config

        if is_train:  # define discriminators
            # specify the training losses you want to print out.
            self.loss_names = ['r_kl_min', 'recon', 'f_kl_max', 'f_kl_min','f_latent_recon']
            self.lambda_r_kl_min = opt['train']['lambda_r_kl_min']
            self.lambda_recon = opt['train']['lambda_recon']
            self.lambda_f_kl_max = opt['train']['lambda_f_kl_max']
            self.lambda_f_kl_min = opt['train']['lambda_f_kl_min']
            self.lambda_latent_recon = opt['train']['lambda_latent_recon']
            self.kl_margin = opt['train'].get('kl_margin', 5.0)

            # define loss functions
            self.criterion_recon = loss.Distance(opt['train']['recon_mode'])
            self.criterion_recon_embed = loss.Distance(opt['train']['recon_embed_mode'])
            self.criterion_kl_min = loss.KLN01Loss('qp', minimize=True)
            self.criterion_kl_max = loss.KLN01Loss('qp', minimize=False)

            # initialize optimizers
            self.optimizer_enc = torch.optim.Adam(self.encoder.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))
            self.optimizer_dec = torch.optim.Adam(self.decoder.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))

            if opt['train'].get('load', False):
                self.load_networks(opt['train'].get('load_suffix', 'latest'))
                print('Network Loaded!')
    
    def set_input(self, data):
        # Reference sample for both pitch and timbre
        self.data = self.resize(data['data'].to(self.device))

    def forward(self):
        """Run forward pass"""
        self.latent = self.encoder(self.data)
        self.recon = self.decoder(self.latent)

    def generate(self, data):
        with torch.no_grad():
            latent = self.encoder(data)
            return self.decoder(latent)

    def resize(self, data):
        current_phase = len(self.encoder.enc_net) - 1
        while current_phase > self.net_phase:
            if current_phase != 7:
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
        self.encoder.config = current_config
        self.decoder.config = current_config

        if has_status_changed:
            self.print_net_status()
    
    def backward_enc(self):
        # minimizing the KL (real_latent, N(0,I))
        self.loss_r_kl_min = self.criterion_kl_min(self.latent) * self.lambda_r_kl_min

        # Reconstruction loss
        self.loss_recon = self.criterion_recon(self.recon, self.data)* self.lambda_recon

        # maximizing the KL(fake_latent, N(0,I))
        f_latent = Variable(torch.FloatTensor(self.latent.size()).to(self.device))
        net_utils.populate_embed(f_latent)
        f_recon = self.decoder(f_latent)
        f_latent_recon = self.encoder(f_recon)
        self.loss_f_kl_max = self.criterion_kl_max(f_latent_recon) * self.lambda_f_kl_max

        # Hinge KL divergence Max(-Bound,  loss_r_kl_min - loss_f_kl_max)
        kl_loss = torch.max(-torch.ones_like(self.loss_f_kl_max).to(self.device)* self.kl_margin, \
                        self.loss_r_kl_min + self.loss_f_kl_max)

        # Compute the total loss and backward the loss
        total_loss = kl_loss + self.loss_recon
        total_loss.backward(retain_graph=True)

    def backward_dec(self):
        # minimizing the KL(fake_latent, N(0,I))
        f_latent = Variable(torch.FloatTensor(self.latent.size()).cuda(self.device))
        net_utils.populate_embed(f_latent)
        f_recon = self.decoder(f_latent)
        f_latent_recon = self.encoder(f_recon)
        self.loss_f_kl_min = self.criterion_kl_min(f_latent_recon) * self.lambda_f_kl_min

        # minimizing the latent distance between fake samples(fake_latent, fake_latent_recon)
        self.loss_f_latent_recon = self.criterion_recon(f_latent_recon, f_latent) * self.lambda_latent_recon

        # Compute the total loss and backward the loss
        total_loss = self.loss_f_kl_min + self.loss_f_latent_recon
        total_loss.backward(retain_graph=True)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()

        self.optimizer_enc.zero_grad() 
        self.backward_enc()
        self.optimizer_enc.step()

        # Updating parameters of the networks
        self.optimizer_dec.zero_grad() 
        self.backward_dec()  
        self.optimizer_dec.step()

        # Progressive Growing parameters update!
        self.update_net_status()