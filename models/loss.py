import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class TCLoss(nn.Module):

    def __init__(self, N, window_size, add_gap = False):
        super(TCLoss, self).__init__()
        self.n_sample = N
        self.window_size = window_size
        self.segment_gap = add_gap
    
    def _sample_anchors(self, B, L, add_gap = False):
        if add_gap:
            segment_length = (L - (self.window_size // 2) * (self.n_sample - 1)) // self.n_sample 
            anchors = np.arange(0, self.n_sample) * (segment_length + self.window_size // 2)
        else:
            segment_length = L // self.n_sample
            anchors = np.arange(0, self.n_sample) * segment_length
        
        anchor_offsets = np.random.randint(0, segment_length, (B, self.n_sample))

        anchors = anchors[None, :] + anchor_offsets
        return anchors

    def _sample_positive(self, anchors, L):

        half_window = self.window_size // 2

        offset_choices = [x for x in range(1, half_window + 1)] + [-x for x in range(1, half_window + 1)] 
        pos_offsets = np.random.choice(offset_choices, (anchors.shape[0], anchors.shape[1]))

        positive = np.clip(anchors + pos_offsets, 0, L-1)

        return positive
    
    @staticmethod
    def sim_matrix(a, b, eps=1e-12):
        a_norm = F.normalize(a, dim = -1)
        b_norm = F.normalize(b, dim = -1)
        sim_mt = torch.matmul(a_norm, b_norm.transpose(-2, -1))
        return sim_mt

    def forward(self, feats, feats_length = None):
        B, L, C = feats.shape
        # [B N]
        anchors = self._sample_anchors(B, L, self.segment_gap)
        positive = self._sample_positive(anchors, L)
        
        anchor_feats = feats[torch.arange(B).unsqueeze(-1), anchors]
        positive_feats = feats[torch.arange(B).unsqueeze(-1), positive]

        sim = self.sim_matrix(anchor_feats, positive_feats)

        label = torch.arange(self.n_sample, device=sim.device).unsqueeze(0).repeat(B, 1)
        loss = F.cross_entropy(sim, label)
        
        return loss


class MaskedTCLoss(nn.Module):

    def __init__(self, N, window_size, add_gap = False):
        super(MaskedTCLoss, self).__init__()
        self.n_sample = N
        self.window_size = window_size
        self.segment_gap = add_gap
    
    def _sample_anchors(self, B, L, add_gap = False):
        L = np.clip(L, 8, None)
        if add_gap:
            segment_length = (L - (self.window_size // 2) * (self.n_sample - 1)) // self.n_sample 
            anchors = np.arange(0, self.n_sample) * (segment_length + self.window_size // 2)
        else:
            segment_length = L // self.n_sample
            anchors = np.arange(0, self.n_sample)[None, : ] * segment_length[:, None] # B N_sample
        # print(segment_length, segment_length.shape)
        highs = [[x] for x in segment_length]
        anchor_offsets = np.random.randint([0] * self.n_sample, highs) # B N_sample

        anchors = anchors + anchor_offsets

        return anchors

    def _sample_positive(self, anchors, L):

        half_window = self.window_size // 2

        offset_choices = [x for x in range(1, half_window + 1)] + [-x for x in range(1, half_window + 1)] 
        pos_offsets = np.random.choice(offset_choices, (anchors.shape[0], anchors.shape[1]))

        B = anchors.shape[0]
        positive = np.clip(anchors + pos_offsets, 0, (L-1)[:, None])

        return positive
    
    @staticmethod
    def sim_matrix(a, b, eps=1e-12):
        a_norm = F.normalize(a, dim = -1)
        b_norm = F.normalize(b, dim = -1)
        sim_mt = torch.matmul(a_norm, b_norm.transpose(-2, -1))
        return sim_mt

    def forward(self, feats, feats_length = None):
        B, L, C = feats.shape
        # [B N]
        anchors = self._sample_anchors(B, feats_length, self.segment_gap)
        positive = self._sample_positive(anchors, feats_length)
        
        anchor_feats = feats[torch.arange(B).unsqueeze(-1), anchors]
        positive_feats = feats[torch.arange(B).unsqueeze(-1), positive]

        sim = self.sim_matrix(anchor_feats, positive_feats)

        label = torch.arange(self.n_sample, device=sim.device).unsqueeze(0).repeat(B, 1)
        loss = F.cross_entropy(sim, label)
        
        return loss