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

class DualAEClassModel(BaseModel):
    """
    """
    def __init__(self, opt, is_train= True):
        """Initialize Dual Autoencoder with timbre and pitch encoder.

        Parameters:
            opt (dict)      - stores all the experiment flags; needs to be a subclass of BaseOptions
            is_train (bool) - Stage flag; {True: Training, False: Testing}
        """
        BaseModel.__init__(self, opt, is_train= True)

        # Instantiating networks
        self.AE = networks.instantiate_net(opt['model']['ae_net'])
        self.AE.to(self.device)
        self.model_names = ['AE']

        if is_train:  # define
            # Specify the training losses you want to print out.
            self.loss_names = ['recon']
            self.lambda_recon = opt['train']['lambda_recon']
            self.lambda_p_class = opt['train'].get('lambda_p_class', 0.0)
            self.lambda_t_class = opt['train'].get('lambda_t_class', 0.0)

            if self.lambda_p_class != 0:        
                self.loss_names += ['p_class']
            if self.lambda_t_class != 0:        
                self.loss_names += ['t_class']

            # Define loss functions
            self.criterion_recon = loss.Distance(opt['train']['recon_mode'])
            self.criterion_entropy = nn.CrossEntropyLoss()

            self.optimizer_AE = torch.optim.Adam(self.AE.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))
            print('Learning Rate: {}'.format(opt['train']['lr']))
            if opt['train'].get('load', False):
                self.load_networks(opt['train'].get('load_suffix', 'latest'))
                print('Network Loaded!')
    
    def set_input(self, data):
        self.data = data['data'].to(self.device)
        self.pitch = torch.argmax(data['pitch'].to(self.device), dim=1, keepdim=False)
        self.instr = torch.argmax(data['instr'].to(self.device), dim=1, keepdim=False)

    def forward(self):
        """Run forward pass"""
        self.recon, self.h_timbre, self.h_pitch, self.pred_timbre, self.pred_pitch = self.AE(self.data)

    def validate(self, data):
        with torch.no_grad():
            self.forward()
            self.compute_losses()

    def compute_losses(self):
        # Reconstruction loss
        self.loss_recon = self.criterion_recon(self.recon, self.data)* self.lambda_recon

        # Pitch loss
        if self.lambda_p_class != 0:        
            self.loss_p_class = self.criterion_entropy(self.pred_pitch, self.pitch)* self.lambda_p_class
        else:
            self.loss_p_class = 0

        # Timbre loss
        if self.lambda_t_class != 0:        
            self.loss_t_class = self.criterion_entropy(self.pred_timbre, self.instr)* self.lambda_t_class
        else:
            self.loss_t_class = 0

    def backward_AE(self):
        self.compute_losses()

        total_loss = self.loss_recon + self.loss_p_class + self.loss_t_class
        total_loss.backward(retain_graph=True)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()

        self.optimizer_AE.zero_grad() 
        self.backward_AE()  
        self.optimizer_AE.step() 