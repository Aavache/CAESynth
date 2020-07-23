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

class DuoEDModel(BaseModel):
    """
    This class defined the generator ,discriminator, siamese and optimizers for a Siamese conditional GAN model as well as implementing the logic for
    optimizing the parameters. The siamese network will be in charge of disentangle the information related to timbre by optimizing a typical triplet 
    loss function. As a condition, the Generator and Discriminator can be global conditioned to a class embedding.
    """

    def __init__(self, opt, is_train= True):
        """Initialize the DuoED Model class.

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
            self.loss_names = ['recon', 'pitch']
            self.lambda_recon = opt['train']['lambda_recon']
            self.lambda_pitch = opt['train']['lambda_pitch']

            # define loss functions
            if opt['train']['recon_mode'] == 'l1':
                self.criterion_recon = nn.L1Loss()
            else:
                self.criterion_recon = nn.MSELoss()
            self.criterion_pitch = nn.CrossEntropyLoss()

            # initialize optimizers
            self.optimizer_E = torch.optim.Adam(self.Enc.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))
            self.optimizer_D = torch.optim.Adam(self.Dec.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))
            
            self.class_exist = opt['model'].get('class', False)
            if self.class_exist:
                self.loss_names += ['class']
                self.lambda_class = opt['train']['lambda_class']

                # Classifier
                self.Class = networks.instantiate_classifier(opt)
                self.Class.to(self.device)
                self.model_names += ['Class']
                self.criterion_class = nn.CrossEntropyLoss()

                self.optimizer_C = torch.optim.Adam(self.Class.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))

            if opt['train'].get('load', False):
                self.load_networks(opt['train'].get('load_suffix', 'latest'))
    
    def set_input(self, data):
        self.pitch = data['pitch']#.to(self.device)
        self.input = data['data']#.to(self.device)

    def forward(self):
        """Run forward pass"""
        self.z_timbre, self.z_pitch = self.Enc(self.input) # Encoding Latent timbre and pitch
        self.recon_data = self.Dec(self.z_timbre, self.z_pitch)

    def generate(self):
        with torch.no_grad():
            self.forward()

    def backward_C(self):
        """Calculate Siamese loss for S"""
        # Pitch classification from latent space of timbre, z_timbre. pitch_from_z_timbre already computed in backward_E
        self.loss_class.backward()
    
    def backward_D(self):
        """Calculate GAN loss for discriminator D"""
        # Reconstruction loss already computed for Encoder
        self.loss_recon.backward()

    def backward_E(self):
        """Calculate the loss for generators G"""
        # Reconstruction loss
        if self.lambda_recon > 0:
            self.loss_recon = self.criterion_recon(self.input, self.recon_data)* self.lambda_recon
        else:
            self.loss_recon = 0

        # Classification loss
        if self.class_exist:
            self.pitch_from_z_timbre = self.Class(self.z_timbre)
            self.loss_class = self.criterion_class(self.pitch_from_z_timbre, self.pitch)* self.lambda_class
        else:
            self.pitch_from_z_timbre = 0
            self.loss_class = 0

        # Pitch loss
        self.loss_pitch = self.criterion_pitch(self.z_pitch, self.pitch)* self.lambda_pitch

        # Combined loss and calculate gradients
        loss_enc_total =  self.loss_recon - self.loss_class + self.loss_pitch
        loss_enc_total.backward(retain_graph=True)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.

        net_utils.set_requires_grad([self.Enc, self.Dec], True)
        self.optimizer_E.zero_grad()   # set S gradients to zero
        self.backward_E()      # calculate gradients for S
        self.optimizer_E.step()  # update S's weights

        net_utils.set_requires_grad([self.Enc], False)  # Gs require no gradients when optimizing Decoder
        self.optimizer_D.zero_grad()  # set Decoder's gradients to zero
        self.backward_D()             # calculate gradients for Decoder
        self.optimizer_D.step()       # update G weights

        #network_utils.set_requires_grad([self.Enc], False)
        if self.class_exist:
            net_utils.set_requires_grad([self.Class], True)
            self.optimizer_C.zero_grad()   # set D gradients to zero
            self.backward_C()      # calculate gradients for D
            self.optimizer_C.step()  # update D's weights