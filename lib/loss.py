import torch
import torch.nn as nn
import torch.nn.functional as F
# internal libs
from lib import util

class TripletLoss(torch.nn.Module):
    def __init__(self, loss_type='l1', alpha = 0.2, hinge=True):
        super(TripletLoss, self).__init__()
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.MSELoss()

        self.alpha = alpha # Positive Number to avoid trivial solution
        self.hinge = hinge

    def forward(self, anchor, positive, negative):
        """
        Computes the triplet loss between an anchor, positive and negative embedding.
        Positive belongs to the anchor domain, while negative does not. When hinged, we lower bound of
        the loss as 0.
        """
        anc_pos_loss = self.loss(anchor, positive)
        anc_neg_loss = self.loss(anchor, negative)
        if self.hinge:
            return F.relu(anc_pos_loss - anc_neg_loss + self.alpha)
        else:
            return anc_pos_loss - anc_neg_loss + self.alpha

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.

    Taken from CycleGAN pytorch Implementation: 
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(self, gan_mode, device, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        self.device = device
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real).to(self.device)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class CosDistance(nn.Module):
    def __init__(self):
        super(CosDistance, self).__init__()
    
    def forward(self, x, y):
        x_norm = util.normalize(x)
        y_norm = util.normalize(y)

        ret = 2 - (x_norm).mul(y_norm)
        return ret.mean(dim=1).mean()
    
class VonMisesDist(nn.Module):
    def __init__(self):
        super(VonMisesDist, self).__init__()
    
    def forward(self, x, y):
        return (-torch.sum(torch.cos(y - x)))/(x.size(0)*x.size(1))

class Distance(nn.Module):
    def __init__(self, mode):
        super(Distance, self).__init__()
        if mode == 'l1':
            self.criterion = nn.L1Loss()
        elif mode == 'l2':
            self.criterion = nn.MSELoss()
        elif mode == 'weighted_l2':
            self.criterion = WeightedMSE()
        elif mode == 'cos':
            self.criterion = CosDistance()
        elif mode == 'von-mises':
            self.criterion = VonMisesDist()
        else:
            raise NotImplementedError

    def forward(self, x, y):
        return self.criterion(x, y)

class WeightedMSE(nn.Module):
    def __init__(self, weight=None):
        super(WeightedMSE, self).__init__()
        self.weights = weight

    def forward(self, x, y):
        weights = self.weights.repeat([x.size(0)] + [1]*len(x.size()[1:])) # Repeat along the batch axis
        return torch.mean(weights*(x - y)**2)

class KLN01Loss(torch.nn.Module): #Adapted from https://github.com/DmitryUlyanov/AGE

    def __init__(self, direction, minimize):
        super(KLN01Loss, self).__init__()
        self.minimize = minimize
        assert direction in ['pq', 'qp'], 'direction?'

        self.direction = direction

    def forward(self, samples):

        assert samples.nelement() == samples.size(1) * samples.size(0), '?'

        samples = samples.view(samples.size(0), -1)

        #samples_var = util.var(samples)
        samples_var = torch.var(samples)
        samples_mean = samples.mean(0)

        if self.direction == 'pq':
            t1 = (1 + samples_mean.pow(2)) / (2 * samples_var.pow(2))
            t2 = samples_var.log()

            KL = (t1 + t2 - 0.5).mean()
        else:
            # In the AGE implementation, there is samples_var^2 instead of samples_var^1
            t1 = (samples_var + samples_mean.pow(2)) / 2
            # In the AGE implementation, this did not have the 0.5 scaling factor:
            t2 = -0.5*samples_var.log()

            KL = (t1 + t2 - 0.5).mean()

        if not self.minimize:
            KL *= -1
        return KL
    
class KLDLoss(torch.nn.Module): #Adapted from https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py
    def __init__(self):
        super(KLDLoss, self).__init__()
        pass

    def forward(self, mu, log_var):
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

class KLD2GaussianLoss(torch.nn.Module):
    ''' KL Divergence between two diagonal Gaussians
    '''
    def __init__(self):
        super(KLD2GaussianLoss, self).__init__()
        pass

    def forward(self, q_mu, q_logvar, p_mu, p_logvar):
        return -0.5 * (1 + q_logvar - p_logvar - (torch.pow(q_mu - p_mu, 2) + torch.exp(q_logvar)) / torch.exp(p_logvar))

def cal_gradient_penalty(DISC, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
        
    Taken from CycleGAN pytorch Implementation: 
             https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
        interpolatesv = real_data
    elif type == 'fake':
        interpolatesv = fake_data
    elif type == 'mixed':
        alpha = torch.rand(real_data.shape[0], 1, device=device)
        alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
        interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
    else:
        raise NotImplementedError('{} not implemented'.format(type))
    interpolatesv = torch.autograd.Variable(interpolatesv, requires_grad=True).to(device)
    disc_interpolates = DISC(interpolatesv)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True)
    gradients = gradients[0].reshape(real_data.size(0), -1)  # flat the data
    gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp # added eps
    return gradient_penalty, gradients