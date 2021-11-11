# External libs
import os
import torch
import torch.nn as nn
# Internal libs
from lib.models.base_model import BaseModel
from lib import networks, loss

class SFCAESynthEnvModel(BaseModel):
    def __init__(self, opt, is_train= True):
        """Initialize a Single-frame CAESynth model trained with musical and environment audio samples.

        Parameters:
            opt (dict)      - stores all the experiment configuration
            is_train (bool) - Stage flag; {True: Training, False: Testing}
        """
        BaseModel.__init__(self, opt, is_train= True)

        # Instantiating networks
        self.AE = networks.instantiate_net(opt['model']['ae_net'])
        self.AE.to(self.device)
        self.model_names = ['AE']

        if is_train:
            # Specify the training losses you want to print out.
            self.loss_names = ['recon']
            self.lambda_recon = opt['train']['lambda_recon']
            self.lambda_t_class = opt['train'].get('lambda_t_class', 0.0)
            self.lambda_p_disc = opt['train'].get('lambda_p_disc', 0.0)

            if self.lambda_t_class != 0:        
                self.loss_names += ['t_class']

            # Whether this model discriminates pitch in the timbre embedding.
            if self.lambda_p_disc != 0.0:   
                self.DISC_P = networks.instantiate_net(opt['model']['pitch_disc'])             
                self.DISC_P.to(self.device)
                self.model_names += ['DISC_P']
                self.loss_names += ['p_disc'] 
                self.optimizer_DISC_P = torch.optim.Adam(self.DISC_P.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))

            self.criterion_recon = loss.Distance(opt['train']['recon_mode'])
            self.criterion_entropy = nn.CrossEntropyLoss()

            if opt['train']['recon_mode'] == 'weighted_l2':
                weight_mse = torch.linspace(10, 1, 1024).unsqueeze(0) #[1, 1024]
                self.criterion_recon.criterion.weights = weight_mse.to(self.device)

            self.optimizer_AE = torch.optim.Adam(self.AE.parameters(), lr=opt['train']['lr'], betas=(opt['train']['beta1'], 0.999))
            print('Learning Rate: {}'.format(opt['train']['lr']))
            if opt['train'].get('load', False):
                self.load_networks(opt['train'].get('load_suffix', 'latest'))
                print('Network Loaded!')
    
    def set_input(self, data):
        self.data = data['nsynth_data'].to(self.device)
        self.one_hot_pitch = data['nsynth_pitch'].to(self.device)
        self.one_hot_pitch = self.one_hot_pitch.repeat([self.data.size(-1),1])
        self.pitch = torch.argmax(self.one_hot_pitch, dim=1, keepdim=False)
        self.instr = torch.argmax(data['nsynth_instr'].to(self.device), dim=1, keepdim=False)
        self.instr = self.instr.repeat(self.data.size(-1))

        self.fsd_data = data['fsd_data'].to(self.device)
        self.fsd_pitch = data['fsd_pitch'].to(self.device)
        self.fsd_pitch = self.fsd_pitch.repeat([self.data.size(-1), 1])
        self.fsd_instr = torch.argmax(data['fsd_instr'].to(self.device), dim=1, keepdim=False)
        self.fsd_instr = self.fsd_instr.repeat(self.data.size(-1))

    def forward(self):
        """Run forward pass"""
        self.recon, self.z_t, self.pred_timbre = self.AE(self.data, pitch=self.one_hot_pitch) # Musical
        self.fsd_recon, _, self.fsd_pred_timbre = self.AE(self.fsd_data, pitch=self.fsd_pitch) # Environment
        if self.lambda_p_disc != 0: # Only musical        
            self.p_in_t = self.DISC_P(self.z_t) # Pitch classification in the timbre code

    def validate(self):
        with torch.no_grad():
            self.forward()
            self.compute_losses()

    def compute_losses(self):
        # Reconstruction loss
        recon = self.recon.squeeze(0).squeeze(0)
        recon = torch.transpose(recon, 0, 1)
        data = self.data.squeeze(0).squeeze(0)
        data = torch.transpose(data, 0, 1)
        self.loss_recon = self.criterion_recon(recon, data)* self.lambda_recon

        fsd_recon = self.fsd_recon.squeeze(0).squeeze(0)
        fsd_recon = torch.transpose(fsd_recon, 0, 1)
        fsd_data = self.fsd_data.squeeze(0).squeeze(0)
        fsd_data = torch.transpose(fsd_data, 0, 1)       
        self.loss_recon += self.criterion_recon(fsd_recon, fsd_data)* self.lambda_recon

        # Pitch discrimination loss in timbre
        if self.lambda_p_disc != 0:        
            self.loss_p_disc = self.criterion_entropy(self.p_in_t, self.pitch)* self.lambda_p_disc
        else:
            self.loss_p_disc = 0

        # Timbre classification loss
        if self.lambda_t_class != 0:        
            self.loss_t_class = self.criterion_entropy(self.pred_timbre, self.instr)* self.lambda_t_class
            self.loss_t_class += self.criterion_entropy(self.fsd_pred_timbre, self.fsd_instr)* self.lambda_t_class
        else:
            self.loss_t_class = 0

    def backward_AE(self):
        self.compute_losses()

        total_loss = self.loss_recon + self.loss_t_class - self.loss_p_disc
        total_loss.backward(retain_graph=True)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()

        self.optimizer_AE.zero_grad() 
        self.backward_AE()  
        self.optimizer_AE.step() 

        if self.lambda_p_disc != 0:   
            self.optimizer_DISC_P.zero_grad() 
            self.loss_p_disc.backward(retain_graph=True)  
            self.optimizer_DISC_P.step()