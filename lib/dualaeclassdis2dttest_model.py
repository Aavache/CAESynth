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

class DualAEClassDis2DtTestModel(BaseModel):
    """
    """
    def __init__(self, opt, is_train= True):
        """Initialize Dual Autoencoder with timbre and pitch encoder as well as Pitch classifier and Discriminator trained with 
        2 datasets, Nsynth(requires timbre and pitch classification) and FSD(wihtout pitch).

        Parameters:
            opt (dict)      - stores all the experiment flags; needs to be a subclass of BaseOptions
            is_train (bool) - Stage flag; {True: Training, False: Testing}
        """
        BaseModel.__init__(self, opt, is_train= True)

        # Instantiating networks
        self.AE = networks.instantiate_net(opt['model']['ae_net'])
        self.AE.to(self.device)
        self.model_names = ['AE']

        if is_train:  # define discriminators
            # Specify the training losses you want to print out.
            self.loss_names = ['recon']
            self.lambda_recon = opt['train']['lambda_recon']
            self.lambda_p_class = opt['train'].get('lambda_p_class', 0.0)
            self.lambda_t_class = opt['train'].get('lambda_t_class', 0.0)
            self.lambda_p_disc = opt['train'].get('lambda_p_disc', 0.0)
            self.lambda_t_disc = opt['train'].get('lambda_t_disc', 0.0)

            if self.lambda_p_class != 0:        
                self.loss_names += ['p_class']
            if self.lambda_t_class != 0:        
                self.loss_names += ['t_class']

            # This network discriminates Pitch in the Timbre embedding.
            if self.lambda_p_disc != 0.0:   
                self.DISC_P = networks.instantiate_net(opt['model']['pitch_disc'])             
                self.DISC_P.to(self.device)
                self.model_names += ['DISC_P']
                self.loss_names += ['p_disc'] 
                self.optimizer_DISC_P = torch.optim.Adam(self.DISC_P.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))


            # This network discriminates Timbre in the Pitch embedding.
            if self.lambda_t_disc != 0.0:   
                self.DISC_T = networks.instantiate_net(opt['model']['timbre_disc'])             
                self.DISC_T.to(self.device)
                self.loss_names += ['t_disc'] 
                self.model_names += ['DISC_T']

                self.optimizer_DISC_T = torch.optim.Adam(self.DISC_T.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))
                
            # Define loss functions
            self.criterion_recon = loss.Distance(opt['train']['recon_mode'])
            self.criterion_entropy = nn.CrossEntropyLoss()

            if opt['train']['recon_mode'] == 'weighted_l2':
                weight_mse = torch.linspace(10, 1, 1024).unsqueeze(0).unsqueeze(0).unsqueeze(3) #[1, 1, 1024, 1]
                weight_mse = weight_mse.repeat((1,1,1,64)).to(self.device)
                self.criterion_recon.criterion.weights = weight_mse

            self.optimizer_AE = torch.optim.Adam(self.AE.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))
            print('Learning Rate: {}'.format(opt['train']['lr']))
            if opt['train'].get('load', False):
                self.load_networks(opt['train'].get('load_suffix', 'latest'))
                print('Network Loaded!')
    
    def set_input(self, data):
        self.nsynth_data = data['nsynth_data'].to(self.device)
        self.pitch = torch.argmax(data['pitch'].to(self.device), dim=1, keepdim=False)
        self.instr = torch.argmax(data['instr'].to(self.device), dim=1, keepdim=False)

        self.fsd_data = data['fsd_data'].to(self.device)
        self.fsd_instr = torch.argmax(data['fsd_instr'].to(self.device), dim=1, keepdim=False)

    def forward(self):
        """Run forward pass"""
        self.nsynth_recon, self.h_timbre, self.h_pitch, self.pred_timbre, self.pred_pitch = self.AE(self.nsynth_data)
        self.fsd_recon, _, self.h_fsd_pitch, self.fsd_pred_timbre, _ = self.AE(self.fsd_data)
        if self.lambda_p_disc != 0:        
            self.p_in_t = self.DISC_P(self.h_timbre) # Pitch classification in Timbre
        if self.lambda_t_disc != 0:        
            self.t_in_p = self.DISC_T(self.h_pitch) # Pitch classification in Timbre
            self.fsd_t_in_p = self.DISC_T(self.h_fsd_pitch)

    def validate(self, data):
        with torch.no_grad():
            self.forward()
            self.compute_losses()

    def compute_losses(self):
        # Reconstruction loss
        self.loss_recon = self.criterion_recon(self.nsynth_recon, self.nsynth_data)* self.lambda_recon
        self.loss_recon += self.criterion_recon(self.fsd_recon, self.fsd_data)* self.lambda_recon

        # Pitch Classification Loss in Timbre
        if self.lambda_p_disc != 0:        
            self.loss_p_disc = self.criterion_entropy(self.p_in_t, self.pitch)* self.lambda_p_disc
        else:
            self.loss_p_disc = 0

        # Pitch loss
        if self.lambda_p_class != 0:        
            self.loss_p_class = self.criterion_entropy(self.pred_pitch, self.pitch)* self.lambda_p_class
        else:
            self.loss_p_class = 0

        # Timbre loss
        if self.lambda_t_class != 0:        
            self.loss_t_class = self.criterion_entropy(self.pred_timbre, self.instr)
            self.loss_t_class += self.criterion_entropy(self.fsd_pred_timbre, self.fsd_instr)* self.lambda_t_class
        else:
            self.loss_t_class = 0

        # Timbre Classification Loss in Pitch
        if self.lambda_t_disc != 0:        
            self.loss_t_disc = self.criterion_entropy(self.t_in_p, self.instr)* self.lambda_t_disc
            self.loss_t_disc += self.criterion_entropy(self.fsd_t_in_p, self.instr)* self.lambda_t_disc
        else:
            self.loss_t_disc = 0

    def backward_AE(self):
        self.compute_losses()

        total_loss = self.loss_recon + self.loss_p_class + self.loss_t_class - self.loss_p_disc - self.loss_t_disc
        total_loss.backward(retain_graph=True)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()

        self.optimizer_AE.zero_grad() 
        self.backward_AE()  
        # torch.nn.utils.clip_grad_norm(self.AE.parameters(), 1)
        self.optimizer_AE.step() 

        if self.lambda_p_disc != 0:   
            self.optimizer_DISC_P.zero_grad() 
            self.loss_p_disc.backward(retain_graph=True)  
            # torch.nn.utils.clip_grad_value_(self.DISC_P.parameters(), 1)
            self.optimizer_DISC_P.step()
        
        if self.lambda_t_disc != 0:   
            self.optimizer_DISC_T.zero_grad() 
            self.loss_t_disc.backward(retain_graph=True)  
            # torch.nn.utils.clip_grad_value_(self.DISC_T.parameters(), 1)
            self.optimizer_DISC_T.step()