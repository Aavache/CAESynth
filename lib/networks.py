# External libs
import sys
import torch
import torch.nn as nn

class MFEncoder(nn.Module):
    ''' 
     Multi-Frame Encoder
    '''
    def __init__(self, in_ch = 1, latent_size=512, alpha=0.2):
        super(MFEncoder, self).__init__()
        layers = []
        layers += [nn.Conv2d(in_ch, 128, (1,1), (1,1)),
                 nn.Conv2d(128, 128, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(128),
                 nn.Conv2d(128, 256, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(256),
                 nn.Conv2d(256, 256, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(256),
                 nn.Conv2d(256, 256, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(256),
                 nn.Conv2d(256, 256, (4,4), (2,1), (1,0)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(256),
                 nn.Conv2d(256, 512, (4,1), (2,1), (1,0)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(512),
                 nn.Conv2d(512, 512, (4,1), (2,1), (1,0)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(512),
                 nn.Conv2d(512, 512, (4,1), (2,1), (1,0)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(512),
                 nn.Conv2d(512, latent_size, (4,1), (1,1))]
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)

class SFEncoder(nn.Module):
    ''' 
     Single-Frame Encoder
    '''
    def __init__(self, in_size = 1024, mid_size = 512, latent_size=64):
        super(SFEncoder, self).__init__()
        layers = []
        layers = [nn.Linear(in_size, mid_size),
                nn.Tanh(),
                nn.Linear(mid_size, mid_size),
                nn.Tanh(),
                nn.Linear(mid_size, mid_size),
                nn.Tanh(),
                nn.Linear(mid_size, mid_size),
                nn.Tanh(),
                nn.Linear(mid_size, mid_size),
                nn.Tanh(),
                nn.Linear(mid_size, latent_size)]
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)

class MFDecoder(nn.Module):
    ''' 
     Multi-Frame Decoder
    '''
    def __init__(self, in_ch = 1, latent_size=1024, alpha=0.2):
        super(MFDecoder, self).__init__()
        layers = []
        layers += [nn.ConvTranspose2d(latent_size, 1024, (4,1), (1,1)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(1024),
                 nn.ConvTranspose2d(1024, 512, (4,1), (2,1), (1,0)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(512),
                 nn.ConvTranspose2d(512, 512, (4,1), (2,1), (1,0)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(512),
                 nn.ConvTranspose2d(512, 256, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(256),
                 nn.ConvTranspose2d(256, 256, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(256),
                 nn.ConvTranspose2d(256, 256, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(256),
                 nn.ConvTranspose2d(256, 256, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(256),
                 nn.ConvTranspose2d(256, 128, (4,4), (2,2), (1,1)),
                 nn.LeakyReLU(alpha),
                 nn.BatchNorm2d(128),
                 nn.ConvTranspose2d(128, 128, (4,4), (2,2), (1,1)),
                 nn.ConvTranspose2d(128, in_ch, (1,1), (1,1))]
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)

class SFDecoder(nn.Module):
    ''' 
     Single-Frame Decoder
    '''
    def __init__(self, in_size = 1024, mid_size = 512, latent_size=64):
        super(SFDecoder, self).__init__()
        layers = []
        layers = [nn.Linear(latent_size, mid_size),
                nn.Tanh(),
                nn.Linear(mid_size, mid_size),
                nn.Tanh(),
                nn.Linear(mid_size, mid_size),
                nn.Tanh(),
                nn.Linear(mid_size, mid_size),
                nn.Tanh(),
                nn.Linear(mid_size, mid_size),
                nn.Tanh(),
                nn.Linear(mid_size, in_size)]
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)

class MFCAESynth(nn.Module):
    ''' 
    Multi-frame CAESynth autoencoder plus timbre classifier. The pitch classifier is defined 
    separately for convinience.
    '''
    def __init__(self, in_ch = 1024, pitch_class=84, timbre_class=28):
        super(MFCAESynth, self).__init__()
        
        self.enc_t = MFEncoder(in_ch, 512)
        self.dec = MFDecoder(in_ch, pitch_class + 512)

        self.class_t = Classifier(512, timbre_class, 512)
    
    def encode(self, input):
        return self.enc_t(input)

    def decode(self, latent):
        return self.dec(latent)

    def forward(self, input, pitch):
        z_t = self.encode(input)
        
        return self.decode(torch.cat([z_t, pitch], dim=1)), z_t, self.class_t(z_t)

class SFCAESynth(nn.Module):
    ''' 
    Single-frame CAESynth autoencoder plus timbre classifier. The pitch classifier is defined 
    separately for convinience.
    '''
    def __init__(self, in_size = 1024, timbre_size=512, pitch_class=84, timbre_class=28, mid_size=512):
        super(SFCAESynth, self).__init__()

        self.enc = SFEncoder(in_size, mid_size, timbre_size)
        self.dec = SFDecoder(in_size, mid_size, timbre_size+pitch_class)
        self.timbre_class = Classifier(timbre_size, timbre_class, mid_size)
    
    def encode(self, input):
        in_data = input.squeeze(0).squeeze(0)
        in_data = torch.transpose(in_data, 0, 1)
        timbre_latent = self.enc(in_data)
        return timbre_latent

    def decode(self, latent):
        out = self.dec(latent)
        return torch.transpose(out, 0, 1).unsqueeze(0).unsqueeze(0)

    def forward(self, input, pitch):
        z_t = self.encode(input)
        out = self.decode(torch.cat([z_t, pitch],dim=1))
        return out, z_t, self.timbre_class(z_t)

class MFBaseline(nn.Module):
    ''' 
        Multi-frame baseline based on https://arxiv.org/pdf/1704.01279.pdf. This net concatenates 
        the pitch to the timbre latent code.
    '''
    def __init__(self, in_ch = 1024, pitch_class=84, timbre_class=28):
        super(MFBaseline, self).__init__()
        self.enc_t = MFEncoder(in_ch, 512)
        self.dec = MFDecoder(in_ch, pitch_class + 512)
    
    def encode(self, input):
        return self.enc_t(input)

    def decode(self, latent):
        return self.dec(latent)

    def forward(self, input, pitch):
        z_t = self.encode(input)
        return self.decode(torch.cat([z_t, pitch], dim=1)), z_t, None

class SFBaseline(nn.Module):
    ''' 
        Single-frame baseline based on https://arxiv.org/pdf/2001.11296.pdf . This net concatenates 
        the pitch label to both the input and the timbre latent code.
    '''
    def __init__(self, in_size = 1024, timbre_size=512, pitch_class=84, timbre_class=28, mid_size=512):
        super(SFBaseline, self).__init__()

        self.enc_t = SFEncoder(in_size + pitch_class, mid_size, timbre_size)
        self.dec = SFDecoder(in_size, mid_size, pitch_class + timbre_size)

    def encode(self, input, pitch):
        in_data = input.squeeze(0).squeeze(0)
        in_data = torch.transpose(in_data, 0, 1)
        in_data = torch.cat([in_data, pitch], dim=1)
        return self.enc_t(in_data)

    def decode(self, latent):
        out = self.dec(latent)
        return torch.transpose(out, 0, 1).unsqueeze(0).unsqueeze(0)

    def forward(self, input, pitch):
        z_t = self.encode(input, pitch)
        return self.decode(torch.cat([z_t, pitch], dim=1)), z_t, None

class Classifier(nn.Module):
    def __init__(self, input_size=1024, output_size=84, mid_size=512):
        super(Classifier, self).__init__()
        net_ly = [] 
        net_ly += [nn.Linear(input_size, mid_size)]
        net_ly += [nn.ReLU()]
        net_ly += [nn.Linear(mid_size, mid_size)]
        net_ly += [nn.ReLU()]
        net_ly += [nn.Linear(mid_size, mid_size)]
        net_ly += [nn.ReLU()]
        net_ly += [nn.Linear(mid_size, output_size)]
        self.net = nn.Sequential(*net_ly)
    
    def forward(self, input):
        while len(input.shape)>2:
            input = input.squeeze(-1)
        return self.net(input)


class NSynthBlock(nn.Module):
    '''To be utilized in future works'''
    def __init__(self, in_ch, out_ch, k_size=(4,4), stride=(2,2), pad=(1,1), mode='enc'):
        super(NSynthBlock, self).__init__()
        if mode == 'enc':
            self.conv = nn.Sequential(
                                nn.Conv2d(in_ch, out_ch, k_size, stride, padding=pad),
                                nn.LeakyReLU(0.2),
                                nn.BatchNorm2d(out_ch))
        else:
            self.conv = nn.Sequential(
                                nn.ConvTranspose2d(in_ch, out_ch, k_size, stride, padding=pad),
                                nn.LeakyReLU(0.2),
                                nn.BatchNorm2d(out_ch))

    def forward(self, input):
        return self.conv(input)

def instantiate_net(opt):
    '''Instanciates the network with parameters according to the the configuration file.'''
    current_module = sys.modules[__name__]
    class_ = getattr(current_module, opt['name'])
    return class_(**opt['params'])