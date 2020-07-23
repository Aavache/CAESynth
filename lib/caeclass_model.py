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

class CAEClassModel(BaseModel):
    """
    """

    def __init__(self, opt, is_train= True):
        """Initialize the Autoencoder with Pitch classifier.

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
            # Classifier
            self.Class = networks.instantiate_classifier(opt)
            self.Class.to(self.device)
            self.model_names += ['Class']
            self.criterion_class = nn.CrossEntropyLoss()

            # Specify the training losses you want to print out.
            self.loss_names = ['recon', 'r_class', 'f_class']
            self.lambda_recon = opt['train']['lambda_recon']
            self.lambda_latent = opt['train']['lambda_latent']
            self.lambda_class = opt['train']['lambda_class']
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
            self.optimizer_C = torch.optim.Adam(self.Class.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))

            if opt['train'].get('load', False):
                self.load_networks(opt['train'].get('load_suffix', 'latest'))
    
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

        self.x_class = self.Class(self.x)
        self.y_class = self.Class(self.y)
        self.yx_class = self.Class(self.yx_recon)
        self.xy_class = self.Class(self.xy_recon)

    def generate(self, data, trg_pitch):
        with torch.no_grad():
            z_timbre = self.Enc(data)
            recon = self.Dec(z_timbre, trg_pitch) # Decode zx with y pitch
            return recon, z_timbre

    def backward_C(self):
        """Calculate loss for Classifier Class"""
        self.loss_r_class = self.criterion_class(self.x_class, torch.argmax(self.x_pitch, dim=1, keepdim=False))* self.lambda_class
        self.loss_r_class+= self.criterion_class(self.y_class, torch.argmax(self.y_pitch, dim=1, keepdim=False))* self.lambda_class
        self.loss_r_class.backward(retain_graph=True)

    def backward_D(self):
        """Calculate loss for Decoder Dec"""
        # Loss already computed for Encoder
        loss_decoder = self.loss_recon +  self.loss_f_class
        loss_decoder.backward(retain_graph=True)

    def backward_E(self):
        """Calculate loss for Encoder Enc"""
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

        if self.lambda_class > 0:
            self.loss_f_class = self.criterion_class(self.yx_class, torch.argmax(self.x_pitch ,dim=1,keepdim=False))* self.lambda_class
            self.loss_f_class += self.criterion_class(self.xy_class, torch.argmax(self.y_pitch ,dim=1,keepdim=False))* self.lambda_class
        else:
            self.loss_class = 0

        # Combined loss and calculate gradients
        loss_enc_total =  self.loss_recon + self.loss_latent + self.loss_f_class
        loss_enc_total.backward(retain_graph=True)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.

        #net_utils.set_requires_grad([self.Enc], False)  #  require no gradients when optimizing Decoder
        self.optimizer_C.zero_grad()  # set classifier's gradients to zero
        self.backward_C()             # calculate gradients for classifier
        self.optimizer_C.step()       # update Class weights

        #net_utils.set_requires_grad([self.Enc, self.Dec, self.Class], True)
        self.optimizer_E.zero_grad()   # set Encoder gradients to zero
        self.backward_E()      # calculate gradients for Encoder
        self.optimizer_E.step()  # update E's weights

        #net_utils.set_requires_grad([self.Enc], False)  # require no gradients when optimizing Decoder
        self.optimizer_D.zero_grad()  # set Decoder's gradients to zero
        self.backward_D()             # calculate gradients for Decoder
        self.optimizer_D.step()       # update D weights
