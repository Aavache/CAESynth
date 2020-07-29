import torch
import torch.nn as nn
import torch.nn.functional as F

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