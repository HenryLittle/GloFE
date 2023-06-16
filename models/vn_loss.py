import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl

def pair_wise_cosine_sim(a, b, dim=-1):
    a_norm = F.normalize(a, p=2, dim=dim)
    b_norm = F.normalize(b, p=2, dim=dim)
    return torch.einsum('...lc, ...nc -> ...ln', a_norm, b_norm)


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


class IntraSampleContrastiveLoss(nn.Module):

    def __init__(
        self, vn_embed_layer, margin, 
        num_head:int=4, visual_dim:int=768, matched_thresh:int=2
    ):
        super(IntraSampleContrastiveLoss, self).__init__()
        self.margin = margin
        
        self.matched_thresh = matched_thresh
        self.vn_embed_layer = vn_embed_layer # nn.embedding for vns (initialized from GloVe vectors, [Vocab, 300])

        self.vn_embed_dim = self.vn_embed_layer.weight.shape[1]
        
        self.xatten = nn.MultiheadAttention(
            embed_dim = self.vn_embed_dim, 
            num_heads = num_head, 
            kdim = visual_dim, vdim = visual_dim,
            batch_first=True
        )
    

    def forward(self, visual_feat, visual_feat_len, vns, vn_len):
        """
        Args:
            visual_feat: [B, L, C]
            visual_feat_len: [B]
            vns: [B, MAX_VN]
            vn_len: [B]

        Returns:
            Intra sample loss between matched vns and visual features
        """
        # filter by number of matched vns
        f_idxs = (vn_len > self.matched_thresh).nonzero().squeeze(-1)
        visual_feat, visual_feat_len, vns, vn_len = visual_feat[f_idxs], visual_feat_len[f_idxs], vns[f_idxs], vn_len[f_idxs]

        # clip max_vn to reduce redundent calculation
        MAX_VN = vn_len.max()
        vns = vns[:, :MAX_VN]
        vn_embeds = self.vn_embed_layer(vns) # [B, MAX_VN, C]

        # query visual features
        padding_mask = gen_padding_mask(visual_feat.shape[0], visual_feat.shape[1], visual_feat_len)
        xatten_out, _ = self.xatten(
            query = vn_embeds,
            key = visual_feat, value = visual_feat,
            key_padding_mask = padding_mask, 
            need_weights = False
        ) # [B, MAX_VN, C]
        # compute scores (cosine similarity) [B, VN_embed, VN_xatten]
        scores = pair_wise_cosine_sim(vn_embeds, xatten_out) # [B, MAX_VN, MAX_VN]
        scores_diag = torch.diagonal(scores, dim1 = -2, dim2 = -1) # [B, MAX_VN]

        # margin loss
        loss = (self.margin + scores - scores_diag.unsqueeze(-1)).clamp(min=0) # [B, MAX_VN, MAX_VN]

        # mask out diagonal and outer rim
        squa_mask = gen_square_mask(vn_len, MAX_VN)
        diag_mask = (torch.eye(MAX_VN) > 0.5).to(visual_feat.device)
        # print(squa_mask.shape, diag_mask.shape, loss.shape)
        masked_loss = loss.masked_fill_(torch.logical_or(squa_mask, diag_mask), 0)

        # different # of samples, normalizing by smaple size ?
        loss_per_sample = masked_loss.sum(dim = (-1, -2)) / (vn_len - 1) ** 2
        loss_out = loss_per_sample.mean()
        return loss_out


class GloVeEmbedding(nn.Embedding):
    def __init__(self, vocab, embed_dim):
        super(GloVeEmbedding, self).__init__(vocab, embed_dim)
        path = '/mnt/workspace/slt_baseline/notebooks/uncased_filtred_glove_VN_embed.pkl'
        with open(path, 'rb') as f:
            glove_np = pkl.load(f)
            assert glove_np.shape[0] == vocab and glove_np.shape[1] == embed_dim
            self.weight = nn.Parameter(torch.from_numpy(glove_np), requires_grad=True)


if __name__ == '__main__':
    B, L, C = 32, 256, 768
    fake_input = torch.randn(B, L, C)
    fake_input_length = torch.randint(20, L, (B,))
    print('input feature shape:', fake_input.shape)

    MAX_VN = 20
    VN_VOCAB = 5563
    VN_EMBED_DIM = 300
    fake_matched_vn = torch.randint(0, VN_VOCAB, (B, MAX_VN))
    fake_matched_vn_length = torch.randint(0, MAX_VN, (B,))
    print(f'fake_matched_vn: {fake_matched_vn.shape}\nfake_matched_vn_length: {fake_matched_vn_length.shape}')

    fake_word_embed = torch.randn(VN_VOCAB, VN_EMBED_DIM)
    print(f'fake_word_embed:{fake_word_embed.shape}')

    ### Intra sample contrastive loss ###
    embed = GloVeEmbedding(VN_VOCAB, VN_EMBED_DIM)
    loss = IntraSampleContrastiveLoss(embed, margin=5.0)

    lout = loss(fake_input, fake_input_length, fake_matched_vn, fake_matched_vn_length)


    print(lout)