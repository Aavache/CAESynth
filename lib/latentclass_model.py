# External libs
import os
import torch
import torch.nn as nn
# Internal libs
from .base_model import BaseModel
from . import networks, loss

class LatentClassModel(BaseModel):
    
    def __init__(self, opt, is_train= True):
        """ Initialize a latent classifier to evaluate the disentanglement

        Parameters:
            opt (dict)      - stores all the experiment flags; needs to be a subclass of BaseOptions
            is_train (bool) - Stage flag; {True: Training, False: Testing}
        """
        BaseModel.__init__(self, opt, is_train= True)

        # Instantiating networks
        self.AE = networks.instantiate_net(opt['model']['class'])
        self.AE.to(self.device)
        load_filename = '%s_%s.pth' % ('latest', 'AE')
        load_path = os.path.join(self.save_dir, load_filename)
        state_dict = torch.load(load_path, map_location=str(self.device))
        self.AE.load_state_dict(state_dict)

        self.classifier = networks.instantiate_net(opt['model']['class'])
        self.classifier.to(self.device)
        self.model_names = ['classifier']
        self.label = opt['label']
        self.mode = opt['mode']

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
        if self.mode == 'sf':
            self.one_hot_pitch = data['pitch'].to(self.device)
            self.one_hot_pitch = self.one_hot_pitch.repeat([self.data.size(-1),1])
            self.pitch = torch.argmax(self.one_hot_pitch, dim=1, keepdim=False)
            self.instr = torch.argmax(data['instr'].to(self.device), dim=1, keepdim=False)
            self.instr = self.instr.repeat(self.data.size(-1))
        else: # mf
            self.pitch = torch.argmax(data['pitch'], dim=1, keepdim=False).to(self.device)
            # self.one_hot_pitch = data['pitch'].to(self.device)
            self.instr = torch.argmax(data['instr'], dim=1, keepdim=False).to(self.device)

    def forward(self):
        """Run forward pass"""
        h = self.AE.encode(self.data, self.one_hot_pitch)
        #h = self.AE.encode(self.data)
        if len(h.size())>2:
            h = h.squeeze(-1).squeeze(-1)
        self.pred = self.classifier(h) # Pitch classification in Timbre

    def validate(self, data):
        with torch.no_grad():
            self.forward()
            self.compute_losses()

    def compute_losses(self):
        # Classification Loss
        if self.label == 'timbre':
            self.loss_class = self.criterion_entropy(self.pred, self.instr)
        else: # Pitch
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
        