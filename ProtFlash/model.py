import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask

class RelativatePositionEmbeddingModule(nn.Module):
    """relativate position embedding

        Args:
            embedding_dim (int): pair relative position embedding dim. Defaults to 128.
            max_rel_dist (int, optional): max relative distance. Defaults to 64.
    """

    def __init__(self, max_rel_dist=32):
        super().__init__()
        self.max_rel_dist = max_rel_dist
        self.pos_embedding = nn.Embedding(max_rel_dist*2 + 1, 1)

    def forward(self, aa_embedding):
        """
        Args:
            seq_length (int): protein seq length

        Returns:
            rel_pos_emb [tensor]: [1, 1, N_seq, N_seq]
        """
        # seq_length-> N_seq
        seq_index = torch.arange(aa_embedding.shape[-1]).to(aa_embedding.device)  # N_seq
        seq_rel_dist = rearrange(seq_index, 'i -> () i ()') - \
            rearrange(seq_index, 'j -> () () j')  # [1, N_seq, N_seq]
        seq_rel_dist = seq_rel_dist.abs().clamp(0, self.max_rel_dist)
        # [1, N_seq, N_seq, embedding_dim]
        rel_pos_emb = self.pos_embedding(seq_rel_dist).squeeze(-1)
        return rel_pos_emb.unsqueeze(0)

def rope(x, axis):
    """RoPE position embedding."""
    shape = x.shape
    if isinstance(axis, int):
        axis = [axis]

    spatial_shape = [shape[i] for i in axis]
    total_len = 1
    for i in spatial_shape:
        total_len *= i
    position = torch.arange(0, total_len, step=1.0).view(spatial_shape).type_as(x)
    for i in range(axis[-1] + 1, len(shape)-1, 1):
        position = position.unsqueeze(-1) #[1, spatial_shape]
    half_size = shape[-1] // 2
    freq_seq = torch.arange(half_size).type_as(x) / float(half_size)
    inv_freq = 10000 ** -freq_seq
    sinusoid = torch.einsum('...,d->...d', position, inv_freq) # [1, spatial_shape, d/2]
    sin = torch.sin(sinusoid)
    cos = torch.cos(sinusoid)
    x1, x2 = torch.split(x, half_size, dim=-1) # [batch, N_seq, d/2]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder


class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = torch.einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)


class MixedChunkAttention(nn.Module):
    def __init__(self, dim, qk_dim, group_size=64, expansion_factor=2, max_rel_dist=32, attn_dropout=0.0):
        super().__init__()
        self.group_size = group_size
        hidden_dim = int(dim * expansion_factor)
        self.to_query_key = nn.Sequential(nn.Linear(dim, qk_dim), nn.SiLU())
        # self.key = nn.Linear(dim, inner_dim, bias=False)
        self.to_value = nn.Sequential(nn.Linear(dim, hidden_dim), nn.SiLU())

        self.to_out = nn.Linear(hidden_dim, dim)
        # pre_layer_norm
        self.pre_norm = nn.LayerNorm(dim)
        self.post_norm = nn.LayerNorm(dim)

        self.gating = nn.Sequential(nn.Linear(dim, hidden_dim), nn.SiLU())

        self.qk_offset_scale = OffsetScale(qk_dim, heads = 4)

        self.rel_pos_bias = RelativatePositionEmbeddingModule(max_rel_dist=max_rel_dist)

        self.dropout = nn.Dropout(attn_dropout)

    
    def forward(self, x, mask=None):
        """
        Args:
            x (`tensor`): [batch, N_seq, embedding_size]
            mask (`tensor`, optional): [batch, N_seq]. Defaults to None.
            chunk_size (`int`, optional):  Defaults to 64.

        Returns:
            out (tensor): [batch, N_seq, num_heads*head_dim]
        """
        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size
        x = self.pre_norm(x)
        qk, v = self.to_query_key(x), self.to_value(x)
        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)

        quad_q = rope(quad_q, axis=1)
        lin_q = rope(lin_q, axis=1)
        quad_k = rope(quad_k, axis=1)
        lin_k = rope(lin_k, axis=1)

        padding = padding_to_multiple_of(n, g)

        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v = map(lambda t: F.pad(t, (0, 0, 0, padding), value = 0.), (quad_q, quad_k, lin_q, lin_k, v))
        if mask is not None:
            mask = F.pad(mask, (0, padding), value=0)
            mask = rearrange(mask, 'b (g j) -> b g () j', j = self.group_size)

        quad_q, quad_k, lin_q, lin_k, v = map(lambda t: rearrange(t, 'b (g n) d -> b g n d', n = self.group_size), (quad_q, quad_k, lin_q, lin_k, v))
        # global
        # global_scale = 1/(self.scale * x.shape[-1] * x.shape[-3])
        quad_att = torch.einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / self.group_size
        quad_att = quad_att + self.rel_pos_bias(quad_att)
        attn = F.relu(quad_att) ** 2
        attn = self.dropout(attn)
        if mask is not None:
            attn = attn.masked_fill(mask==0, 0.)
        
        quad_out = torch.einsum('... i j, ... j d -> ... i d', attn, v)

        lin_kv = torch.einsum('b g n d, b g n e -> b d e', lin_k, v) / n
        lin_out = torch.einsum('b g n d, b d e -> b g n e', lin_q, lin_kv)

        quad_attn_out, lin_attn_out = map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n], (quad_out, lin_out))

        gating_x = self.gating(x)
        out = gating_x * (quad_attn_out + lin_attn_out)
        out = self.post_norm(self.to_out(out))
        return out


class FLASHTransformer(nn.Module):
    def __init__(
        self,
        dim,
        num_tokens,
        depth,
        group_size = 64,
        query_key_dim = 128,
        expansion_factor = 2.,
        attn_dropout = 0.,
        max_rel_dist=32,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(num_tokens+2, dim)
        self.group_size = group_size

        self.layers = nn.ModuleList([MixedChunkAttention(dim, query_key_dim, group_size, expansion_factor, max_rel_dist, attn_dropout) for _ in range(depth)])

    def forward(
        self,
        x,
        lengths = None
    ): 
        if lengths is not None:
            mask = length_to_mask(lengths).to(x.device)
        else:
            mask = None
        x = self.token_emb(x)
        for sublayer in self.layers:
            x = sublayer(x, mask = mask) + x
        return x


