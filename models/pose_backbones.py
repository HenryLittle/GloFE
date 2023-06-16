
import sys
sys.path.append('..')
from models.ctrgcn_base_p76 import Model as PoseBackbone

from torch import nn
import torch.nn.functional as F
import numpy as np
import torch

from einops import rearrange


class PoseBackboneWrapper(nn.Module):
    def __init__(self):
        super(PoseBackboneWrapper, self).__init__()
        self.pose_model = PoseBackbone(
                num_class=2000, num_point=76, num_person=1, 
                graph='models.graph.mmpose_p76.Graph',
                graph_args={'labeling_mode': 'spatial'}, drop_out=0)
        pose_weights = torch.load('/mnt/workspace/slt_baseline/models/ckpt/ctr_76_e1/runs-88-100144.pt')
        # output [B C T V]
        self.pose_model.load_state_dict(pose_weights, strict=False)
        self.feat_dim = self.pose_model.fc.in_features
    
    def forward(self, prefix):
        pose_output = self.pose_model(prefix) # B C T V
        pose_pool = pose_output.mean(-1) # B C T
        prefix = pose_pool.transpose(-1, -2) # B T C
        return prefix


class SlidingWindowPoseBackbone(nn.Module):
    
    def __init__(self):
        super(SlidingWindowPoseBackbone, self).__init__()
        self.pose_model = PoseBackbone(
                num_class=2000, num_point=76, num_person=1, 
                graph='models.graph.mmpose_p76.Graph',
                graph_args={'labeling_mode': 'spatial'}, drop_out=0)
        pose_weights = torch.load('/mnt/workspace/slt_baseline/models/ckpt/ctr_76_e1/runs-88-100144.pt')
        # output [B C T V]
        self.pose_model.load_state_dict(pose_weights, strict=False)
        self.feat_dim = self.pose_model.fc.in_features
    
    @staticmethod
    def gen_slide(length, span=8, step=2):
        if length <= span:
            diff = span - length
            idxs = np.array(range(length))
            idxs = np.concatenate((idxs, (length-1)*np.ones(diff)))
        else:
            num_clips = (length - span + (step - 1)) // step + 1
            offsets = np.arange(num_clips)[:,None] * step
            idxs = offsets + np.arange(span)[None, :]
        # ensure within range
        idxs = np.clip(idxs, 0, length - 1)
        return idxs


    def forward(self, prefix):
        B, _, T, _ = prefix.shape

        slide_index = self.gen_slide(T, span = 8, step=6)
        prefix_slide = prefix[:, :, slide_index, :] # B C W 8 V
        prefix_slide = rearrange(prefix_slide, 'B C W S V -> (B W) C S V').contiguous()
        pose_output = self.pose_model(prefix_slide) # BW C T V
        
        pose_pool = pose_output.mean(-1).mean(-1) # BW C 

        pose_pool = rearrange(pose_pool, '(B W) C -> B W C', B = B)

        return pose_pool


class TSWPartedPoseBackbone(nn.Module):
    
    def __init__(self):
        super(TSWPartedPoseBackbone, self).__init__()
        self.pose_model = PoseBackbone(
                num_class=2000, num_point=76, num_person=1, 
                graph='models.graph.mmpose_p76.Graph',
                graph_args={'labeling_mode': 'spatial'}, drop_out=0)
        pose_weights = torch.load('/mnt/workspace/slt_baseline/models/ckpt/ctr_76_e1/runs-88-100144.pt')
        # output [B C T V]
        self.pose_model.load_state_dict(pose_weights, strict=False)
        # self.feat_dim = self.pose_model.fc.in_features
        self.feat_dim = 128 * 4
        # reduce total layers
        self.pose_model = nn.Sequential(
            self.pose_model.l1,
            self.pose_model.l2,
            self.pose_model.l3,
            self.pose_model.l4,
            self.pose_model.l5,
        )
    
    @staticmethod
    def gen_slide(length, span=8, step=2):
        if length <= span:
            diff = span - length
            idxs = np.array(range(length))
            idxs = np.concatenate((idxs, (length-1)*np.ones(diff)))
        else:
            num_clips = (length - span + (step - 1)) // step + 1
            offsets = np.arange(num_clips)[:,None] * step
            idxs = offsets + np.arange(span)[None, :]
        # ensure within range
        idxs = np.clip(idxs, 0, length - 1)
        return idxs


    def forward(self, prefix):
        B, _, T, _ = prefix.shape

        slide_index = self.gen_slide(T, span = 8, step=6)
        prefix_slide = prefix[:, :, slide_index, :] # B C W 8 V
        prefix_slide = rearrange(prefix_slide, 'B C W S V -> (B W) C S V').contiguous()
        pose_output = self.pose_model(prefix_slide) # BW C T V
        
        # pose_pool = pose_output.mean(-1).mean(-1) # BW C 
        # Pool respective body regions
        body_pool = pose_output[:, :, :, :11].mean(-1, keepdim=True)
        hand_l = pose_output[:, :, :, 11:32].mean(-1, keepdim=True)
        hand_r = pose_output[:, :, :, 32:53].mean(-1, keepdim=True)
        face = pose_output[:, :, :, 53:].mean(-1, keepdim=True)

        pose_pool = torch.cat((body_pool, hand_l, hand_r, face), dim=-1) # BW C T 4
        pose_pool = pose_pool.mean(-2) # BW C 4

        pose_pool = rearrange(pose_pool, '(B W) C P -> B W (P C)', B = B, P = 4)

        return pose_pool



class PartedPoseBackbone(nn.Module):
    def __init__(self):
        super(PartedPoseBackbone, self).__init__()
        self.pose_model = PoseBackbone(
                num_class=2000, num_point=76, num_person=1, 
                graph='models.graph.mmpose_p76.Graph',
                graph_args={'labeling_mode': 'spatial'}, drop_out=0)
        pose_weights = torch.load('models/ckpt/ctr_76_e1/runs-88-100144.pt')
        # output [B C T V]
        self.pose_model.load_state_dict(pose_weights, strict=False)
        self.feat_dim = self.pose_model.fc.in_features * 4
    

    def forward(self, prefix):
        pose_output = self.pose_model(prefix) # B C T V

        # Pool respective body regions
        body_pool = pose_output[:, :, :, :11].mean(-1, keepdim=True)
        hand_l = pose_output[:, :, :, 11:32].mean(-1, keepdim=True)
        hand_r = pose_output[:, :, :, 32:53].mean(-1, keepdim=True)
        face = pose_output[:, :, :, 53:].mean(-1, keepdim=True)

        pose_pool = torch.cat((body_pool, hand_l, hand_r, face), dim=-1) # B C T 4

        pose_pool = rearrange(pose_pool, 'B C T P -> B T (P C)')

        return pose_pool


class OPPartedPoseBackbone(nn.Module):
    def __init__(self):
        super(OPPartedPoseBackbone, self).__init__()
        self.pose_model = PoseBackbone(
                num_class=2000, num_point=78, num_person=1, 
                graph='models.graph.openpose_78.Graph',
                graph_args={'labeling_mode': 'spatial'}, drop_out=0)
        pose_weights = torch.load('models/ckpt/ctr_op78_mix_HF05_F64_e1/runs-82-93316.pt')
        # output [B C T V]
        self.pose_model.load_state_dict(pose_weights, strict=False)
        self.feat_dim = self.pose_model.fc.in_features * 4
    

    def forward(self, prefix):
        pose_output = self.pose_model(prefix) # B C T V

        # Pool respective body regions
        body_pool = pose_output[:, :, :, :11].mean(-1, keepdim=True)
        hand_l = pose_output[:, :, :, 11:32].mean(-1, keepdim=True)
        hand_r = pose_output[:, :, :, 32:53].mean(-1, keepdim=True)
        face = pose_output[:, :, :, 53:].mean(-1, keepdim=True)

        pose_pool = torch.cat((body_pool, hand_l, hand_r, face), dim=-1) # B C T 4

        pose_pool = rearrange(pose_pool, 'B C T P -> B T (P C)')

        return pose_pool