# External libs
import os
import torch
import torch.nn as nn
# Internal libs
from .base_model import BaseModel
from . import networks, loss

class MFCAESynthEnvModel(BaseModel):
    """
    """
    def __init__(self, opt, is_train= True):
        """Initialize Multi-frame CAESynth model trained with musical and environment audio dataset.

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
            self.lambda_t_class = opt['train'].get('lambda_t_class', 0.0)
            self.lambda_p_disc = opt['train'].get('lambda_p_disc', 0.0)
            
            if self.lambda_t_class != 0:        
                self.loss_names += ['t_class']

            # This network discriminates Pitch in the Timbre embedding.
            if self.lambda_p_disc != 0.0:   
                self.DISC_P = networks.instantiate_net(opt['model']['pitch_disc'])             
                self.DISC_P.to(self.device)
                self.model_names += ['DISC_P']
                self.loss_names += ['p_disc'] 
                self.optimizer_DISC_P = torch.optim.Adam(self.DISC_P.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))

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
        self.data = data['nsynth_data'].to(self.device)
        self.one_hot_pitch = data['nsynth_pitch'].to(self.device)
        self.pitch = torch.argmax(self.one_hot_pitch, dim=1, keepdim=False)
        self.instr = torch.argmax(data['nsynth_instr'].to(self.device), dim=1, keepdim=False)

        self.fsd_data = data['fsd_data'].to(self.device)
        self.fsd_pitch = data['fsd_pitch'].to(self.device)
        # self.fsd_instr = data['fsd_instr'].to(self.device)

    def forward(self):
        """Run forward pass"""
        self.recon, self.h_timbre, self.pred_timbre = self.AE(self.data, pitch=self.one_hot_pitch)
        self.fsd_recon = self.AE(self.fsd_data)[0]
        if self.lambda_p_disc != 0:        
            self.p_in_t = self.DISC_P(self.h_timbre) # Pitch classification in Timbre

    def validate(self, data):
        with torch.no_grad():
            self.forward()
            self.compute_losses()

    def compute_losses(self):
        # Reconstruction loss
        self.loss_recon = self.criterion_recon(self.recon, self.data)* self.lambda_recon
        self.loss_recon += self.criterion_recon(self.fsd_recon, self.fsd_data)* self.lambda_recon

        # Pitch Classification Loss in Timbre
        if self.lambda_p_disc != 0:        
            self.loss_p_disc = self.criterion_entropy(self.p_in_t, self.pitch)* self.lambda_p_disc
        else:
            self.loss_p_disc = 0

        # Timbre loss
        if self.lambda_t_class != 0:        
            self.loss_t_class = self.criterion_entropy(self.pred_timbre, self.instr)* self.lambda_t_class
        else:
            self.loss_t_class = 0

    def backward_AE(self):
        self.compute_losses()

        total_loss = self.loss_recon + self.loss_t_class - self.loss_p_disc
        total_loss.backward(retain_graph=True)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()

        self.optimizer_AE.zero_grad() 
        self.backward_AE()  
        self.optimizer_AE.step() 

        if self.lambda_p_disc != 0:   
            self.optimizer_DISC_P.zero_grad() 
            self.loss_p_disc.backward(retain_graph=True)  
            self.optimizer_DISC_P.step()
        