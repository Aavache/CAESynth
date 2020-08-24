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
from . import loss

class AESkipTripletModel(BaseModel):
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
            self.loss_names = ['recon', 'pitch_trip', 'timbre_trip', 'skip_pitch_trip', 'skip_timbre_trip']
            self.lambda_recon = opt['train']['lambda_recon']
            self.lambda_pitch_trp = opt['train']['lambda_pitch_trp']
            self.lambda_skip_pitch_trp = opt['train']['lambda_sk_pitch_trp']
            self.lambda_timbre_trp = opt['train']['lambda_timbre_trp']
            self.lambda_skip_timbre_trp = opt['train']['lambda_sk_timbre_trp']

            # define loss functions
            if opt['train']['recon_mode'] == 'l1':
                self.criterion_recon = nn.L1Loss()
            else:
                self.criterion_recon = nn.MSELoss()

            self.criterion_triplet = loss.TripletLoss()
            # initialize optimizers
            self.optimizer_AE = torch.optim.Adam(self.AE.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))
            self.optimizer_E = torch.optim.Adam(self.AE.enc_net_params, lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))

            if opt['train'].get('load', False):
                self.load_networks(opt['train'].get('load_suffix', 'latest'))
                print('Network Loaded!')
    
    def set_input(self, data):
        # Reference sample for both pitch and timbre
        self.anchor = data['anc_data'].to(self.device)
        # Positive pitch negative timbre sample
        self.dip1 = data['dip1_data'].to(self.device) 
        # Positive timbre negative pitch sample
        self.dip2 = data['dip2_data'].to(self.device) 

    def forward(self):
        """Run forward pass"""
        self.z_anc, self.skip_anc = self.AE.encode(self.anchor)
        self.recon_anc = self.AE.decode(self.z_anc, self.skip_anc)

        self.z_dip1, self.skip_dip1 = self.AE.encode(self.dip1)
        self.recon_dip1 = self.AE.decode(self.z_dip1, self.skip_dip1)

        self.z_dip2, self.skip_dip2 = self.AE.encode(self.dip2) 
        self.recon_dip2 = self.AE.decode(self.z_dip2, self.skip_dip2)

    def generate(self, data):
        with torch.no_grad():
            recon,  z = self.AE(data)
            return recon, z
    
    def backward_AE(self):
        # Reconstruction loss
        self.loss_recon = self.criterion_recon(self.recon_anc, self.anchor)* self.lambda_recon
        self.loss_recon += self.criterion_recon(self.recon_dip1, self.dip1)* self.lambda_recon
        self.loss_recon += self.criterion_recon(self.recon_dip2, self.dip2)* self.lambda_recon

        self.loss_recon.backward(retain_graph=True)

    def backward_E(self):
        # Triple on the Skip Connections
        self.loss_skip_pitch_trip = 0
        self.loss_skip_timbre_trip = 0
        for skip_anc, skip_dip1, skip_dip2 in zip(self.skip_anc, self.skip_dip1, self.skip_dip2):
            anc_tim, anc_pitch = torch.chunk(skip_anc, chunks=2, dim=1)
            dip1_tim, dip1_pitch = torch.chunk(skip_dip1, chunks=2, dim=1)
            dip2_tim, dip2_pitch = torch.chunk(skip_dip2, chunks=2, dim=1)

            # Triple Pitch Loss
            self.loss_skip_pitch_trip += self.criterion_triplet(anc_pitch, dip1_pitch, dip2_pitch)* self.lambda_skip_pitch_trp
            # Triple Timbre Loss
            self.loss_skip_timbre_trip += self.criterion_triplet(anc_tim, dip2_tim, dip1_tim)* self.lambda_skip_timbre_trp 

        # Triple latent Space
        anc_tim, anc_pitch = torch.chunk(self.z_anc, chunks=2, dim=1)
        dip1_tim, dip1_pitch = torch.chunk(self.z_dip1, chunks=2, dim=1)
        dip2_tim, dip2_pitch = torch.chunk(self.z_dip2, chunks=2, dim=1)

        # Triple Pitch Loss
        self.loss_pitch_trip = self.criterion_triplet(anc_pitch, dip1_pitch, dip2_pitch)* self.lambda_pitch_trp
        # Triple Timbre Loss
        self.loss_timbre_trip = self.criterion_triplet(anc_tim, dip2_tim, dip1_tim)* self.lambda_timbre_trp 
                                
        total_loss = self.loss_timbre_trip + self.loss_pitch_trip + self.loss_skip_pitch_trip + \
                        self.loss_skip_timbre_trip 

        total_loss.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()

        self.optimizer_AE.zero_grad() 
        self.backward_AE()  
        self.optimizer_AE.step()

        self.optimizer_E.zero_grad() 
        self.backward_E()  
        self.optimizer_E.step() 