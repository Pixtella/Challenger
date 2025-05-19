import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Linear
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch import Tensor
from typing import Optional, Tuple



class VolumetricPositionEncoding(nn.Module):
    # Relative positional encoding (ROPE inspired rotary pe)

    def __init__(self, feature_dim):
        super().__init__()
        self.feat_dim = feature_dim
        self.vol_bnds = [[0., 0., 0.], [1., 1., 1.]]
        self.voxel_size = 0.08
        self.vol_origin = self.vol_bnds[0]
        self.pe_type = 'rotary'

    def voxelize(self, xyz):
        '''
        @param xyz: B,N,3
        @return: B,N,3
        '''
        if type(self.vol_origin) == list:
            self.vol_origin = torch.FloatTensor(
                self.vol_origin
            ).view(1, 1, -1).to(xyz.device)
        return (xyz - self.vol_origin) / self.voxel_size

    @staticmethod
    def embed_rotary(x, cos, sin):
        '''
        @param x: [B,N,d]
        @param cos: [B,N,d]  [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        @param sin: [B,N,d]  [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        @return:
        '''
        x2 = torch.stack(
            [-x[..., 1::2], x[..., ::2]], dim=-1
        ).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    @staticmethod
    def embed_pos(pe_type, x, pe):
        """Combine feature and position code."""
        if pe_type == 'rotary':
            return VolumetricPositionEncoding.embed_rotary(
                x, pe[..., 0], pe[..., 1]
            )
        elif pe_type == 'sinusoidal':
            return x + pe
        else:
            raise KeyError()

    def forward(self, XYZ):
        '''
        @param XYZ: [B,N,3]
        @return: [B, N, F, 2]
        '''
        original_shape = XYZ.shape
        bsize = original_shape[0]
        XYZ = XYZ.reshape(bsize, -1, original_shape[-1])
        npoint = XYZ.shape[1]

        # bsize, npoint, _ = XYZ.shape
        vox = XYZ
        # vox = self.voxelize(XYZ) # no need voxelize for partnet for now
        x_position, y_position, z_position = (
            vox[..., :1], vox[..., 1:2], vox[..., 2:3]
        )
        div_term = torch.exp(
            torch.arange(0, self.feat_dim // 3, 2, device=XYZ.device).float()
            * (-math.log(10000.0) / (self.feat_dim // 3))
        )
        div_term = div_term.view(1, 1, -1)  # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term)  # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        if self.pe_type == 'sinusoidal':
            position_code = torch.cat([sinx, cosx, siny, cosy, sinz, cosz], -1)

        elif self.pe_type == "rotary":
            # sin/cos [θ0,θ1,θ2......θd/6-1]
            # -> sin/cos [θ0,θ0,θ1,θ1,θ2,θ2......θd/6-1,θd/6-1]
            sinx, cosx, siny, cosy, sinz, cosz = map(
                lambda feat: torch.stack([feat, feat], dim=-1).view(
                    bsize, npoint, -1
                ),
                [sinx, cosx, siny, cosy, sinz, cosz]
            )
            sin_pos = torch.cat([sinx, siny, sinz], dim=-1)
            cos_pos = torch.cat([cosx, cosy, cosz], dim=-1)
            position_code = torch.stack([cos_pos, sin_pos], dim=-1)

        else:
            raise KeyError()

        if position_code.requires_grad:
            position_code = position_code.detach()
        position_code = position_code.reshape(*original_shape[:-1], self.feat_dim, 2)

        return position_code


class MultiheadCustomAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None, slot_competition=False, return_kv=False, gate_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        ##### Custom 
        self.slot_competition = slot_competition
        self.return_kv = return_kv
        self.gate_attn = None
        if gate_attn:
            self.gate_attn = Parameter(torch.randn(num_heads)) #randn
        #####
        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, k_mem=None, v_mem=None, mem_mask=None, rotary_pe=None):
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if hasattr(self, '_qkv_same_embed_dim') and self._qkv_same_embed_dim is False:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, slot_competition=self.slot_competition,
                return_kv = self.return_kv, k_mem = k_mem, v_mem = v_mem,
                gate_attn=self.gate_attn, mem_mask=mem_mask,
                rotary_pe=rotary_pe)
        else:
            if not hasattr(self, '_qkv_same_embed_dim'):
                warnings.warn('A new version of MultiheadAttention module has been implemented. \
                    Please re-train your model with the new module',
                              UserWarning)

            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, slot_competition=self.slot_competition,
                return_kv = self.return_kv, k_mem = k_mem, v_mem = v_mem,
                gate_attn=self.gate_attn, mem_mask=mem_mask,
                rotary_pe=rotary_pe)


def multi_head_attention_forward(query,  # type: Tensor
                                 key,  # type: Tensor
                                 value,  # type: Tensor
                                 embed_dim_to_check,  # type: int
                                 num_heads,  # type: int
                                 in_proj_weight,  # type: Tensor
                                 in_proj_bias,  # type: Tensor
                                 bias_k,  # type: Optional[Tensor]
                                 bias_v,  # type: Optional[Tensor]
                                 add_zero_attn,  # type: bool
                                 dropout_p,  # type: float
                                 out_proj_weight,  # type: Tensor
                                 out_proj_bias,  # type: Tensor
                                 training=True,  # type: bool
                                 key_padding_mask=None,  # type: Optional[Tensor]
                                 need_weights=True,  # type: bool
                                 attn_mask=None,  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,  # type: Optional[Tensor]
                                 k_proj_weight=None,  # type: Optional[Tensor]
                                 v_proj_weight=None,  # type: Optional[Tensor]
                                 static_k=None,  # type: Optional[Tensor]
                                 static_v=None,  # type: Optional[Tensor]
                                 slot_competition=False,
                                 rotary_pe=None,
                                 return_kv = False,
                                 k_mem = None,
                                 v_mem = None,
                                 gate_attn = None,
                                 mem_mask = None
                                 ):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in differnt forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if use_separate_proj_weight is not True:
        if qkv_same:
            # self-attention
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif kv_same:
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                       torch.zeros((attn_mask.size(0), 1),
                                                   dtype=attn_mask.dtype,
                                                   device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    if rotary_pe is not None:  # rotary pe ROPE disentangeld
        qp, kvp = rotary_pe
        q_cos, q_sin = qp[..., 0], qp[..., 1]
        k_cos, k_sin = kvp[..., 0], kvp[..., 1]
        q = VolumetricPositionEncoding.embed_rotary(q.transpose(0, 1), q_cos, q_sin).transpose(0, 1)
        k = VolumetricPositionEncoding.embed_rotary(k.transpose(0, 1), k_cos, k_sin).transpose(0, 1)

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
                                                          dtype=attn_mask.dtype,
                                                          device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                               dtype=key_padding_mask.dtype,
                                               device=key_padding_mask.device)], dim=1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        # attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    if slot_competition:
        attn_output_weights = F.softmax(attn_output_weights, dim=-2) + 1e-8
        attn_output_weights = attn_output_weights / attn_output_weights.sum(dim=-1, keepdim=True)
    else:
        attn_output_weights = F.softmax(
            attn_output_weights, dim=-1)
        
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    
    #do memorizing transformer gating 
    if (gate_attn is not None) and (k_mem is not None) and (v_mem is not None):
        k_mem = k_mem.permute((2, 0, 1))
        key_mem_len = k_mem.shape[0]
        k_mem = k_mem.contiguous().view(key_mem_len, bsz * num_heads, head_dim).transpose(0, 1)
        v_mem = v_mem.permute((2, 0, 1))
        v_mem = v_mem.contiguous().view(key_mem_len, bsz * num_heads, head_dim).transpose(0, 1)
#         if True:
#             k_mem = F.normalize(k_mem, dim = -1)
        
        attn_output_weights_mem = torch.bmm(q, k_mem.transpose(1, 2)) #[24, 16, 110]
        #bcz correspondance b/w key key is good not query, key visually
#         attn_output_weights_mem = torch.bmm(k, k_mem.transpose(1, 2))
        attn_output_weights_mem = F.softmax(attn_output_weights_mem, dim=-1)
        if mem_mask is not None:
            mem_mask = mem_mask.unsqueeze(1).unsqueeze(1) #[B, 1, 1, key_mem_len]
            attn_output_weights_mem = attn_output_weights_mem.reshape(bsz, num_heads, tgt_len, key_mem_len)
            attn_output_weights_mem = attn_output_weights_mem * mem_mask
            attn_output_weights_mem = attn_output_weights_mem.reshape(bsz*num_heads, tgt_len, key_mem_len)
        
        attn_output_weights_mem = F.dropout(attn_output_weights_mem, p=dropout_p, training=training)
        attn_output_mem = torch.bmm(attn_output_weights_mem, v_mem)  #[bsz * num_heads, tgt_len, head_dim]
        
        #gated learnable attention like memorizing transformers
        print("gate_attn ", torch.sigmoid(gate_attn))
        gate = torch.sigmoid(gate_attn).reshape(-1, 1, 1, 1) # (n_head, 1, 1, 1)
        attn_output_mem = attn_output_mem.view(bsz, num_heads, tgt_len, head_dim).transpose(0, 1) #[num_heads, bsz, tgt_len, head_dim]
        attn_output = attn_output.view(bsz, num_heads, tgt_len, head_dim).transpose(0, 1) #[num_heads, bsz, tgt_len, head_dim]
        attn_output = gate*attn_output_mem + (1.-gate)*attn_output
        attn_output = attn_output.transpose(1,0).view(bsz * num_heads, tgt_len, head_dim)
    
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    
    if return_kv:
        return attn_output, q, k, v
    elif need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
#         return attn_output, attn_output_weights.sum(dim=1) / num_heads
        return attn_output, attn_output_weights
    else:
        return attn_output, None

class VerletStandardizer(nn.Module):
    """
    Standardizes trajectories with Verlet-parameterized actions
    See Section 3.2 in https://arxiv.org/pdf/1905.01296
    """
    
    def __init__(self):
        super().__init__()

        self.max_dist = 4   # magic

    def transform_features(self, ego_agent_features, trajectory):
        trajectory = trajectory.reshape(trajectory.shape[0], -1, 3)
        history = ego_agent_features[:,-2:,:3]

        # Apply Verlet parameterization
        full_trajectory = torch.cat([history, trajectory], dim=1)
        deltas = torch.diff(full_trajectory, dim=1)[:,:-1]
        pred_trajectory = full_trajectory[:,1:-1] + deltas
        actions = full_trajectory[:,2:] - pred_trajectory

        # Standardize actions
        actions = actions / self.max_dist

        actions = actions.reshape(actions.shape[0], -1)
        return actions
        
    def untransform_features(self, ego_agent_features, actions):
        actions = actions.reshape(actions.shape[0], -1, 3)
        history = ego_agent_features[:,-2:,:3]
        
        # Unstandardize actions
        actions = actions * self.max_dist

        # Use Verlet parameterization to calculate trajectory
        states = [history[:,0], history[:,1]]
        for t in range(actions.shape[1]):
            states.append((2 * states[-1]) - states[-2] + actions[:,t])
        trajectory = torch.stack(states[2:], dim=1)

        trajectory = trajectory.reshape(trajectory.shape[0], -1)
        return trajectory


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RotaryPositionEncoding(nn.Module):
    def __init__(self, feature_dim, pe_type='Rotary1D'):
        super().__init__()
        self.feature_dim = feature_dim
        self.pe_type = pe_type
    @staticmethod
    def embed_rotary(x, cos, sin):
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x
    def forward(self, x_position):
        bsize, npoint = x_position.shape
        div_term = torch.exp(
            torch.arange(0, self.feature_dim, 2, dtype=torch.float, device=x_position.device)
            * (-math.log(10000.0) / (self.feature_dim)))
        div_term = div_term.view(1, 1, -1) # [1, 1, d]
        sinx = torch.sin(x_position[...,None] * div_term)  # [B, N, d]
        cosx = torch.cos(x_position[...,None] * div_term)
        sin_pos, cos_pos = map(
            lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
            [sinx, cosx]
        )
        position_code = torch.stack([cos_pos, sin_pos] , dim=-1)
        if position_code.requires_grad:
            position_code = position_code.detach()
        return position_code

class ParallelAttentionLayer(nn.Module):
    """Self-/Cross-attention between two sequences."""

    def __init__(self, d_model=256, dropout=0.1, n_heads=8, pre_norm=False,
                 self_attention1=True, self_attention2=True,
                 cross_attention1=True, cross_attention2=True,
                 apply_ffn=True,
                 slot_attention12=False, slot_attention21=False,
                 rotary_pe=False):
        """Initialize layers, d_model is the encoder dimension."""
        super().__init__()
        self.pre_norm = pre_norm
        self.self_attention1 = self_attention1
        self.self_attention2 = self_attention2
        self.cross_attention1 = cross_attention1
        self.cross_attention2 = cross_attention2
        self.apply_ffn = apply_ffn
        self.rotary_pe = rotary_pe

        # Self-attention for seq1
        if self.self_attention1:
            self.sa1 = MultiheadCustomAttention(
                d_model, n_heads, dropout=dropout
            )
            self.dropout_1 = nn.Dropout(dropout)
            self.norm_1 = nn.LayerNorm(d_model)

        # Self-attention for seq2
        if self.self_attention2:
            self.sa2 = MultiheadCustomAttention(
                d_model, n_heads, dropout=dropout
            )
            self.dropout_2 = nn.Dropout(dropout)
            self.norm_2 = nn.LayerNorm(d_model)

        # Cross attention from seq1 to seq2
        self.norm_12 = None
        if cross_attention1:
            self.cross_12 = MultiheadCustomAttention(
                d_model, n_heads, dropout=dropout,
                slot_competition=slot_attention12
            )
            self.dropout_12 = nn.Dropout(dropout)
            self.norm_12 = nn.LayerNorm(d_model)

        # Cross attention from seq2 to seq1
        self.norm_21 = None
        if cross_attention2:
            self.cross_21 = MultiheadCustomAttention(
                d_model, n_heads, dropout=dropout,
                slot_competition=slot_attention21
            )
            self.dropout_21 = nn.Dropout(dropout)
            self.norm_21 = nn.LayerNorm(d_model)

        # FFN-1
        if self_attention1 or cross_attention1:
            self.ffn_12 = nn.Sequential(
                nn.Linear(d_model, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, d_model),
                nn.Dropout(dropout)
            )
            self.norm_122 = nn.LayerNorm(d_model)

        # FFN-2
        if self_attention2 or cross_attention2:
            self.ffn_21 = nn.Sequential(
                nn.Linear(d_model, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, d_model),
                nn.Dropout(dropout)
            )
            self.norm_212 = nn.LayerNorm(d_model)

    def _norm(self, x, layer, normalize=True):
        if normalize and layer is not None:
            return layer(x)
        return x

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, seq1, seq1_key_padding_mask, seq2,
                seq2_key_padding_mask,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=None, seq2_sem_pos=None,
                attn_mask_11=None, attn_mask_22=None,
                attn_mask_12=None, attn_mask_21=None):
        """Forward pass, seq1 (B, S1, F), seq2 (B, S2, F)."""
        rot_args = {}

        # Create key, query, value for seq1, seq2
        q1 = k1 = v1 = self._norm(seq1, self.norm_12, self.pre_norm)
        q2 = k2 = v2 = self._norm(seq2, self.norm_21, self.pre_norm)
        if not self.rotary_pe:
            q1 = k1 = self.with_pos_embed(seq1, seq1_pos)
            q2 = k2 = self.with_pos_embed(seq2, seq2_pos)
        q1 = self.with_pos_embed(q1, seq1_sem_pos)
        k1 = self.with_pos_embed(k1, seq1_sem_pos)
        q2 = self.with_pos_embed(q2, seq2_sem_pos)
        k2 = self.with_pos_embed(k2, seq2_sem_pos)

        # Cross-attention from seq1 to seq2
        if self.cross_attention1:
            if self.rotary_pe:
                rot_args['rotary_pe'] = (seq1_pos, seq2_pos)
            seq1b = self.cross_12(
                query=q1.transpose(0, 1),
                key=k2.transpose(0, 1),
                value=v2.transpose(0, 1),
                attn_mask=attn_mask_12,
                key_padding_mask=seq2_key_padding_mask,  # (B, S2)
                **rot_args,
            )[0].transpose(0, 1)
            seq1 = seq1 + self.dropout_12(seq1b)
            seq1 = self._norm(seq1, self.norm_12, not self.pre_norm)

        # Cross-attention from seq2 to seq1
        if self.cross_attention2:
            if self.rotary_pe:
                rot_args['rotary_pe'] = (seq2_pos, seq1_pos)
            seq2b = self.cross_21(
                query=q2.transpose(0, 1),
                key=k1.transpose(0, 1),
                value=v1.transpose(0, 1),
                attn_mask=attn_mask_21,
                key_padding_mask=seq1_key_padding_mask,  # (B, S1)
                **rot_args,
            )[0].transpose(0, 1)
            seq2 = seq2 + self.dropout_21(seq2b)
            seq2 = self._norm(seq2, self.norm_21, not self.pre_norm)

        # Self-attention for seq1
        if self.self_attention1:
            q1 = k1 = v1 = self._norm(seq1, self.norm_1, self.pre_norm)
            if self.rotary_pe:
                rot_args['rotary_pe'] = (seq1_pos, seq1_pos)
            else:
                q1 = k1 = self.with_pos_embed(seq1, seq1_pos)
            q1 = self.with_pos_embed(q1, seq1_sem_pos)
            k1 = self.with_pos_embed(k1, seq1_sem_pos)
            seq1b = self.sa1(
                query=q1.transpose(0, 1),
                key=k1.transpose(0, 1),
                value=v1.transpose(0, 1),
                attn_mask=attn_mask_11,
                key_padding_mask=seq1_key_padding_mask,  # (B, S1)
                **rot_args,
            )[0].transpose(0, 1)
            seq1 = seq1 + self.dropout_1(seq1b)
            seq1 = self._norm(seq1, self.norm_1, not self.pre_norm)

        # Self-attention for seq2
        if self.self_attention2:
            q2 = k2 = v2 = self._norm(seq2, self.norm_2, self.pre_norm)
            if self.rotary_pe:
                rot_args['rotary_pe'] = (seq2_pos, seq2_pos)
            else:
                q2 = k2 = self.with_pos_embed(seq2, seq2_pos)
            q2 = self.with_pos_embed(q2, seq2_sem_pos)
            k2 = self.with_pos_embed(k2, seq2_sem_pos)
            seq2b = self.sa2(
                query=q2.transpose(0, 1),
                key=k2.transpose(0, 1),
                value=v2.transpose(0, 1),
                attn_mask=attn_mask_22,
                key_padding_mask=seq2_key_padding_mask,  # (B, S2)
                **rot_args,
            )[0].transpose(0, 1)
            seq2 = seq2 + self.dropout_2(seq2b)
            seq2 = self._norm(seq2, self.norm_2, not self.pre_norm)

        # FFN-1
        if (self.self_attention1 or self.cross_attention1) and self.apply_ffn:
            seq1 = self._norm(seq1, self.norm_122, self.pre_norm)
            seq1 = seq1 + self.ffn_12(seq1)
            seq1 = self._norm(seq1, self.norm_122, not self.pre_norm)

        # FFN-2
        if (self.self_attention2 or self.cross_attention2) and self.apply_ffn:
            seq2 = self._norm(seq2, self.norm_212, self.pre_norm)
            seq2 = seq2 + self.ffn_21(seq2)
            seq2 = self._norm(seq2, self.norm_212, not self.pre_norm)

        return seq1, seq2
