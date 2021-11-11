# External libs
import os
import torch
import torch.nn as nn
# Internal libs
from lib.models.base_model import BaseModel
from lib import networks

class LatentClassModel(BaseModel):
    def __init__(self, opt, is_train= True):
        """ Initialize a latent classifier to evaluate latent pitch classification (LPA) of the pretrained model.
        This model can be reused to either train a latent timbre classifier or a latent pitch classifier. In addition
        it is designed to operate in both single and multi frame, to be specified in the config file.

        Parameters:
            opt (dict)      - stores all the experiment configuration
            is_train (bool) - Stage flag; {True: Training, False: Testing}
        """
        BaseModel.__init__(self, opt, is_train= True)

        # Instantiating networks and loading the pretrained model
        self.AE = networks.instantiate_net(opt['model']['class'])
        self.AE.to(self.device)
        load_filename = '%s_%s.pth' % ('latest', 'AE')
        load_path = os.path.join(self.save_dir, load_filename)
        state_dict = torch.load(load_path, map_location=str(self.device))
        self.AE.load_state_dict(state_dict)

        # Instantiating the evaluation networks 
        self.classifier = networks.instantiate_net(opt['model']['class'])
        self.classifier.to(self.device)
        self.model_names = ['classifier']
        self.label = opt['label']
        self.mode = opt['mode']

        if is_train:
            # Specify the training losses you want to print out.
            self.loss_names = ['class']
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
        z_t = self.AE.encode(self.data, self.one_hot_pitch)
        if len(z_t.size())>2:
            z_t = z_t.squeeze(-1).squeeze(-1)
        self.pred = self.classifier(z_t) # Pitch classification in the timbre code

    def validate(self):
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
        self.forward()

        self.optimizer.zero_grad() 
        self.backward()  
        self.optimizer.step()
        