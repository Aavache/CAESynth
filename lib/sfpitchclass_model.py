# External libs
import os
import torch
import torch.nn as nn
# Internal libs
from .base_model import BaseModel
from . import networks, loss

class SFPitchClass(BaseModel):
    """
    """
    def __init__(self, opt, is_train= True):
        """Initialize a single-frame pitch classifier model.

        Parameters:
            opt (dict)      - stores all the experiment flags; needs to be a subclass of BaseOptions
            is_train (bool) - Stage flag; {True: Training, False: Testing}
        """
        BaseModel.__init__(self, opt, is_train= True)

        # Instantiating networks
        self.p_classifier = networks.instantiate_net(opt['model']['class_net'])
        self.p_classifier.to(self.device)
        self.model_names = ['p_classifier']

        if is_train: 
            # Specify the training losses you want to print out.
            self.loss_names = ['p_class']
            self.lambda_p_class = opt['train'].get('lambda_p_class', 0.0)

            self.criterion_entropy = nn.CrossEntropyLoss()

            self.optimizer = torch.optim.Adam(self.p_classifier.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))
            print('Learning Rate: {}'.format(opt['train']['lr']))
            if opt['train'].get('load', False):
                self.load_networks(opt['train'].get('load_suffix', 'latest'))
                print('Network Loaded!')
    
    def set_input(self, data):
        self.data = data['data'].to(self.device)
        one_hot_pitch = data['pitch'].to(self.device)
        one_hot_pitch = one_hot_pitch.repeat([self.data.size(-1),1])
        self.pitch = torch.argmax(one_hot_pitch, dim=1, keepdim=False)

    def forward(self):
        """Run forward pass"""
        self.pred = self.p_classifier(self.data)

    def validate(self, data):
        with torch.no_grad():
            self.forward()
            self.compute_losses()

    def compute_losses(self):
        # Pitch Classification Loss in Timbre
        self.loss_p_class = self.criterion_entropy(self.pred, self.pitch)* self.lambda_p_class

    def backward(self):
        self.compute_losses()

        self.loss_p_class.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()

        self.optimizer.zero_grad() 
        self.backward()  
        self.optimizer.step() 