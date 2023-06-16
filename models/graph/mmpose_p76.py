import sys
import numpy as np

sys.path.extend(['../'])
from models.graph import tools
# from .tools import get_spatial_graph, get_multiscale_spatial_graph

num_node = 76
self_link = [(i, i) for i in range(num_node)]

upper_body_joint_count = 11
hand_joint_count = 21
upper_body = [(0, 1), (0, 2), (2, 4), (1, 3), (0, 5), (0, 6), (6, 8), (8, 10), (5, 7), (7, 9)]
hand = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
# shift hand by 13 Index[13, 33] -> 21 points
hand_0 = [(i + upper_body_joint_count, j + upper_body_joint_count) for (i, j) in hand] # left 
# shift hand_0 by 21 Index[34, 54] -> 21 points
hand_1 = [(i + hand_joint_count, j + hand_joint_count) for (i, j) in hand_0] # right

# mouth_4p = [(0, 2), (2, 1), (0, 3), (3, 1)]
# mouth_4p = [(0, 2), (2, 1), (1, 3), (3, 0)] # cycle
face_23p = [(3, 0), (0, 2), (3, 1), (2, 2)] + \
           [(4, 5), (5, 6), (6, 22)] + [(9, 8), (8, 7), (7, 22)] + \
           [(10, 11), (11, 12), (12, 13), (10, 15), (15, 14), (14, 13), (13, 22)] + \
           [(19, 18), (18, 17), (17, 16), (19, 20), (20, 21), (21, 16), (16, 22)]
offset = upper_body_joint_count + hand_joint_count * 2
face_23p = [(i + offset, j + offset) for (i, j) in face_23p]
inward = upper_body + hand_0 + hand_1 + face_23p

# add link hand to body
# inward += [(4, 13), (7, 34)] # mis-connect but got better results
# Pose[0:11] Hand_L[11:32] Hand_R[32:53]
inward += [(9, 11), (10, 32)]

# inward += [(9, 32), (10, 11)] # mis
# inward += [(13, 34)] # link hand
# inward += [(7, 13), (4, 34)]

# link mouth to nose
inward += [(0, 55)]
inward += [(0, 75)]

outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'multi_scale_spatial':
            A = tools.get_multiscale_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
