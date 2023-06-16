import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl

def pair_wise_cosine_sim(a, b, dim=-1):
    a_norm = F.normalize(a, p=2, dim=dim)
    b_norm = F.normalize(b, p=2, dim=dim)
    return torch.einsum('...lc, ...nc -> ...ln', a_norm, b_norm)

def batched_cosine_sim(a, b, dim=-1):
    a_norm = F.normalize(a, p=2, dim=dim).unsqueeze(-2)
    b_norm = F.normalize(b, p=2, dim=dim).unsqueeze(-1)
    return torch.einsum('...rc, ...cr -> ...r', a_norm, b_norm)

def gen_square_mask(lens, max_len):
    """
        lens: [B], length of each sample
        return: [B, MAX_LEN, MAX_LEN] square mask
    """
    B = lens.shape[0]
    mask = torch.cumsum(torch.ones(B, max_len), dim = 1).to(lens.device) # [B, MAX_LEN]
    mask = mask > lens.unsqueeze(-1)
    mask = torch.logical_or(mask.unsqueeze(-1), mask.unsqueeze(-2))
    return mask.to(lens.device)


def gen_padding_mask(
    B, max_len, lengths
) -> torch.Tensor:

    ones = torch.ones(B, max_len).to(lengths.device) # [B, max_length]
    padding_mask = lengths.type(torch.long).unsqueeze(1) < ones.cumsum(dim=1) # [B, max_length]

    return padding_mask

def gen_batch_ids(B, L, device):
    idxs = torch.arange(0, B).unsqueeze(1).repeat(1, L).to(device)
    return idxs

class InterSampleContrastiveLoss(nn.Module):
    def __init__(self, vn_embed_layer, margin, matched_thresh=2):
        super(InterSampleContrastiveLoss, self).__init__()
        self.margin = margin
        self.matched_thresh = matched_thresh
        self.vn_embed_layer = vn_embed_layer # nn.embedding for vns (initialized from GloVe vectors, [Vocab, 300])

        self.vn_embed_dim = self.vn_embed_layer.weight.shape[1]
        
        self.xatten = nn.MultiheadAttention(
            embed_dim = self.vn_embed_dim, 
            num_heads = 4, 
            kdim = 768, vdim = 768,
            batch_first=True
        )
    

    def forward(self, visual_feat, visual_feat_len, vns, vn_len):
        """
            visual_feat: [B, L, C]
            vns: [B, MAX_VN]
            vn_len: [B]
        """
        # filter by number of matched vns
        f_idxs = (vn_len > self.matched_thresh).nonzero().squeeze(-1)
        visual_feat, visual_feat_len, vns, vn_len = visual_feat[f_idxs], visual_feat_len[f_idxs], vns[f_idxs], vn_len[f_idxs]
        
        # clip max_vn to reduce redundent calculation
        MAX_VN = vn_len.max()
        vns = vns[:, :MAX_VN]

        vn_mask = gen_padding_mask(vns.shape[0], vns.shape[1], vn_len) # [B, MAX_VN]
        masked_vns = vns.masked_fill_(vn_mask, -1) # [B, MAX_VN]
        # filter out VN if it exists in every sample in batch
        t_bvns, t_bvn_count = torch.unique(masked_vns, return_counts=True) # [Batch_VNS], [Batch_VNS]
        t_bvn_out = t_bvns[t_bvn_count >= vn_mask.shape[0]]
        
        if t_bvn_out.shape[0] > 0:
            for vn_out in t_bvn_out:
                if vn_out == -1: continue
                out_idxs = (masked_vns == vn_out)
                masked_vns[out_idxs] = -1
                vn_mask[out_idxs] = True

        batch_vns, bvn_inverse = torch.unique(masked_vns, return_inverse=True) # [Batch_VNS], [B, MAX_VN]
        if batch_vns[0] == -1:
            batch_vns = batch_vns[1:] # drop padding -1
            bvn_inverse = bvn_inverse - 1
        
        bvn_ids = gen_batch_ids(vns.shape[0], vns.shape[1], vns.device) # [B, MAX_VN]
        pos_coord = torch.stack((bvn_inverse, bvn_ids), dim = 2) # [B, MAX_VN, 2], (bvn_id, sample_id)
        pos_coord = pos_coord[~vn_mask] # [Pos_sample_pairs, 2]

        bvn_sample_mask = torch.zeros(batch_vns.shape[0], vns.shape[0]) # [Batch_VNS, B]
        bvn_sample_mask[pos_coord[:, 0], pos_coord[:, 1]] = 1 # [Batch_VNS, B]

        padding_mask = gen_padding_mask(visual_feat.shape[0], visual_feat.shape[1], visual_feat_len)
        batch_vn_embed = self.vn_embed_layer(batch_vns) # [Batch_VNS, C]
        # not sure if it works
        batch_vn_embed_ex = batch_vn_embed.unsqueeze(0).expand(visual_feat.shape[0], -1, -1) # [B, Batch_VNS, C]
        xatten_out, _ = self.xatten(
            query = batch_vn_embed_ex,
            key = visual_feat, value = visual_feat,
            key_padding_mask = padding_mask, 
            need_weights = False
        ) # [B, Batch_VNS, C]
        xatten_out = torch.transpose(xatten_out, 0, 1) # [Batch_VNS, B, C]

        pos_idx = self.random_from_mask(bvn_sample_mask)
        neg_idx = self.random_from_mask(1 - bvn_sample_mask)

        pos_xatten_out = xatten_out[pos_idx[:, 0], pos_idx[:, 1]] # [Batch_VNS, C]
        neg_xatten_out = xatten_out[neg_idx[:, 0], neg_idx[:, 1]] # [Batch_VNS, C]
        pos_score = batched_cosine_sim(batch_vn_embed, pos_xatten_out) # [Batch_VNS, 1]
        neg_score = batched_cosine_sim(batch_vn_embed, neg_xatten_out) # [Batch_VNS, 1]

        loss = (self.margin + neg_score.mean() - pos_score.mean()).clamp(min=0)

        return loss


    def random_from_mask(self, a, num_per_row=1):
        """Randomly select valid elemnts from the boolean[0/1] mask by row
        """
        valid_idx = (a!=0).nonzero()
        unique_rows = valid_idx[:, 0].unique()
        valid_row_idx = [valid_idx[valid_idx[:, 0] == u] for u in unique_rows]
    
        ret = []
        for v in valid_row_idx:
            choice = torch.multinomial(torch.ones(v.size(0)).float(), num_per_row)
            # ret.append(a[v[choice].squeeze().chunk(2)])
            ret.append(v[choice].squeeze())
        ret = torch.stack(ret)
        return ret

class GloVeEmbedding(nn.Embedding):
    def __init__(self, vocab, embed_dim, path, requires_grad=True):
        super(GloVeEmbedding, self).__init__(vocab, embed_dim)
        # path = '/mnt/workspace/slt_baseline/notebooks/uncased_filtred_glove_VN_embed.pkl'
        with open(path, 'rb') as f:
            glove_np = pkl.load(f)
            assert glove_np.shape[0] == vocab and glove_np.shape[1] == embed_dim
            self.weight = nn.Parameter(torch.from_numpy(glove_np), requires_grad=requires_grad)

if __name__ == '__main__':
    B, L, C = 32, 256, 768
    fake_input = torch.randn(B, L, C)
    fake_input_length = torch.randint(20, L, (B,))
    print('input feature shape:', fake_input.shape)

    MAX_VN = 20
    VN_VOCAB = 5563
    VN_EMBED_DIM = 300
    fake_matched_vn = torch.randint(0, VN_VOCAB, (B, MAX_VN))
    # edge case 1: if one VN is contained in all the samples
    fake_matched_vn[:,0] = 10
    fake_matched_vn_length = torch.randint(0, MAX_VN, (B,))
    print(f'fake_matched_vn: {fake_matched_vn.shape}\nfake_matched_vn_length: {fake_matched_vn_length.shape}')

    fake_word_embed = torch.randn(VN_VOCAB, VN_EMBED_DIM)
    print(f'fake_word_embed:{fake_word_embed.shape}')

    ### Intra sample contrastive loss ###
    path = '/mnt/workspace/slt_baseline/notebooks/uncased_filtred_glove_VN_embed.pkl'
    embed = GloVeEmbedding(VN_VOCAB, VN_EMBED_DIM, path)
    loss = InterSampleContrastiveLoss(embed, 0.2)

    lout = loss(fake_input, fake_input_length, fake_matched_vn, fake_matched_vn_length)
    print(lout)
    lout = loss(fake_input, fake_input_length, fake_matched_vn, fake_matched_vn_length)
    print(lout)
    
