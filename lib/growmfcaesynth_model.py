# External libs
import os
import torch
import torch.nn as nn
# Internal libs
from .base_model import BaseModel
from . import networks, loss

class ProgDualAEClassDisModel(BaseModel):
    """
    Progressive growing Autoencoder model
    """
    def __init__(self, opt, is_train= True):
        """Initialize Dual Autoencoder Model.

        Parameters:
            opt (dict)      - stores all the experiment flags; needs to be a subclass of BaseOptions
            is_train (bool) - Stage flag; {True: Training, False: Testing}
        """
        BaseModel.__init__(self, opt, is_train= True)

        # Instantiating networks
        self.AE = networks.instantiate_net(opt['model']['ae_net'])
        self.AE.to(self.device)
        self.model_names = ['AE']

        self.max_phase =  len(self.AE.enc_t.enc_net)-1
        self.fadein_iters = opt['train']['fadein_iters']
        self.stable_iters = opt['train']['stable_iters']
        self.iters = 0

        self.net_status = "stable" # {stable: last trained block is stable , fadein: we are introducing a new block}
        self.net_alpha = 1.0
        if is_train:
            self.net_phase = 0 # the phase denotes the number of conv blocks from the latent space.
        else:
            self.net_phase = self.max_phase

        current_config = {'phase': self.net_phase, 'alpha': self.net_alpha, 'status':self.net_status}
        self.AE.update_config(current_config)

        if is_train:  # define discriminators
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

            self.has_status_changed = True

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
        self.data = self.resize(data['data'].to(self.device))
        self.pitch = torch.argmax(data['pitch'].to(self.device), dim=1, keepdim=False)
        self.instr = torch.argmax(data['instr'].to(self.device), dim=1, keepdim=False)

    def forward(self):
        """Run forward pass"""
        self.recon, self.h_timbre, self.h_pitch, self.pred_timbre, self.pred_pitch = self.AE(self.data)
        if self.lambda_p_disc != 0:        
            self.p_in_t = self.DISC_P(self.h_timbre) # Pitch classification in Timbre
        if self.lambda_t_disc != 0:        
            self.t_in_p = self.DISC_T(self.h_pitch) # Pitch classification in Timbre

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

        # Timbre Classification Loss in Pitch
        if self.lambda_t_disc != 0:        
            self.loss_t_disc = self.criterion_entropy(self.t_in_p, self.instr)* self.lambda_t_disc
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
            self.loss_t_disc.backward()  
            # torch.nn.utils.clip_grad_value_(self.DISC_T.parameters(), 1)
            self.optimizer_DISC_T.step()

        # Progressive Growing parameters update!
        self.update_net_status()
    
    def resize(self, data):
        current_phase = self.max_phase
        while current_phase > self.net_phase:
            if current_phase != self.max_phase:
                data = net_utils.down_sample2x2(data)
            current_phase -= 1
        
        if self.has_status_changed:
            weight_mse = torch.linspace(10, 1, data.size(2)).unsqueeze(0).unsqueeze(0).unsqueeze(3) #[1, 1, 1024, 1]
            weight_mse = weight_mse.repeat((1,1,1,data.size(3))).to(self.device)
            self.criterion_recon.criterion.weights = weight_mse
        return data.detach()
    
    def print_net_status(self):
        print("Net status has changed to: status: {}, phase: {}, alpha: {}".format(self.net_status, \
                        self.net_phase, self.net_alpha))

    def update_net_status(self):
        self.iters += 1
        self.has_status_changed = False
        if self.net_phase == self.max_phase:
            if self.net_status == "fadein":
                if self.iters % self.fadein_iters == 0:
                    self.net_status = "stable"
                    self.has_status_changed = True
        elif self.net_phase == 0:
            if self.iters % self.stable_iters == 0:
                self.net_status = "fadein"
                self.net_phase += 1
                self.has_status_changed = True
        else:
            if self.net_status == "stable":
                if self.iters % self.stable_iters == 0:
                    self.net_status = "fadein"
                    self.net_phase += 1
                    self.has_status_changed = True

            else: # "fadein" 
                if self.iters % self.fadein_iters == 0:
                    self.net_status = "stable"
                    self.has_status_changed = True
        
        if self.net_status == "fadein":
            fadein_perc = (self.iters % self.fadein_iters)/self.fadein_iters
            self.net_alpha = 1 - fadein_perc
        else:
            self.net_alpha = 1

        current_config = {'phase': self.net_phase, 'alpha': self.net_alpha, 'status':self.net_status}
        self.AE.update_config(current_config)

        if self.has_status_changed:
            self.print_net_status()
