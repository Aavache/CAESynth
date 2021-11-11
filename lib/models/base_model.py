# External libs
import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
# Internal libs
from lib import util

class BaseModel(ABC):
    """
    This abstract class serves as a template class for custom model implemenations. 
    """

    def __init__(self, opt, is_train= True):
        """ Initialize the Base Model class.

        Parameters:
            opt (dict)      - stores all the experiment configuration
            is_train (bool) - Stage flag; {True: Training, False: Testing}   
        """
        self.opt = opt 
        self.phase = 'train' if is_train else 'test'
        self.gpu_id = opt[self.phase]['device']
        # get device name: CPU or GPU
        self.device = torch.device('cuda:{}'.format(self.gpu_id)) if self.gpu_id >= 0 else torch.device('cpu')  
        if is_train:
             # save checkpoints in save_dir
            self.save_dir = os.path.join(opt[self.phase]['checkpoints_dir'], opt['experiment_name'], 'snap') 
            util.mkdir(self.save_dir)

        self.loss_names = []
        self.model_names = []
        self.optimizers = []

    @abstractmethod
    def forward(self):
        """Run forward pass"""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights"""
        pass

    def get_current_losses(self):
        """Return traning losses"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) - if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) - current epoch, used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_%s.pth' % (epoch, name)
                if not os.path.isdir(self.save_dir):
                    os.mkdir(self.save_dir)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, name)

                torch.save(net.state_dict(), save_path)

    def load_networks(self, epoch, models=None):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) - current epoch, used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        model_name_list = self.model_names if models is None else models
        for name in model_name_list:
            if isinstance(name, str):
                load_filename = '%s_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                net.load_state_dict(state_dict)

    def eval(self):
        """Switch model to eval mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def train(self):
        """Switch model to train mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()
