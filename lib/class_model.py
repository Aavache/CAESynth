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

class ClassModel(BaseModel):
    """
    """

    def __init__(self, opt, is_train= True):
        """Initialize the Autoencoder with Pitch classifier.

        Parameters:
            opt (dict)      - stores all the experiment flags; needs to be a subclass of BaseOptions
            is_train (bool) - Stage flag; {True: Training, False: Testing}
        """
        BaseModel.__init__(self, opt, is_train= True)
        # Classifier
        self.Class = networks.instantiate_classifier(opt)
        self.Class.to(self.device)
        self.model_names += ['Class']
        self.criterion_class = nn.CrossEntropyLoss()

        if is_train:  # define discriminators
            # Specify the training losses you want to print out.
            self.loss_names = ['class']
            self.lambda_class = opt['train']['lambda_class']

            # initialize optimizers
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
        self.x_class = self.Class(self.x)
        self.y_class = self.Class(self.y)

    def backward_C(self):
        """Calculate loss for Classifier Class"""
        self.loss_class = self.criterion_class(self.x_class, torch.argmax(self.x_pitch, dim=1, keepdim=False))* self.lambda_class
        self.loss_class+= self.criterion_class(self.y_class, torch.argmax(self.y_pitch, dim=1, keepdim=False))* self.lambda_class
        self.loss_class.backward(retain_graph=True)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        
        #net_utils.set_requires_grad([self.Enc], False)  #  require no gradients when optimizing Decoder
        self.optimizer_C.zero_grad()  # set classifier's gradients to zero
        self.backward_C()             # calculate gradients for classifier
        self.optimizer_C.step()       # update Class weights
