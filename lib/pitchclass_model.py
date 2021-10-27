# External libs
import os
import torch
import torch.nn as nn
# Internal libs
from .base_model import BaseModel
from . import networks, loss

class PitchClassModel(BaseModel):
    
    def __init__(self, opt, is_train= True):
        """Initialize a Pitch classifier to evaluate pitch control

        Parameters:
            opt (dict)      - stores all the experiment flags; needs to be a subclass of BaseOptions
            is_train (bool) - Stage flag; {True: Training, False: Testing}
        """
        BaseModel.__init__(self, opt, is_train= True)

        # Instantiating networks
        self.classifier = networks.instantiate_net(opt['model']['class'])
        self.classifier.to(self.device)
        self.model_names = ['classifier']

        if is_train:  # define discriminators
            # Specify the training losses you want to print out.
            self.loss_names = ['class']

            # This network discriminates Pitch in the Timbre embedding.
            self.criterion_entropy = nn.CrossEntropyLoss()

            self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))
            print('Learning Rate: {}'.format(opt['train']['lr']))
            if opt['train'].get('load', False):
                self.load_networks(opt['train'].get('load_suffix', 'latest'))
                print('Network Loaded!')
    
    def set_input(self, data):
        self.data = data['data'].to(self.device)
        self.pitch = torch.argmax(data['pitch'], dim=1, keepdim=False).to(self.device)

    def forward(self):
        """Run forward pass"""
        self.pred = self.classifier(self.data).squeeze(-1).squeeze(-1)

    def validate(self, data):
        with torch.no_grad():
            self.forward()
            self.compute_losses()

    def compute_losses(self):
        # Classification Loss
        self.loss_class = self.criterion_entropy(self.pred, self.pitch)

    def backward(self):
        self.compute_losses()
        self.loss_class.backward(retain_graph=True)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()

        self.optimizer.zero_grad() 
        self.backward()  
        self.optimizer.step()
        