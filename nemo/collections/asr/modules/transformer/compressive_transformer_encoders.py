# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import List, Optional, Set
from inspect import isfunction

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig

from mogrifier import Mogrifier

import math
from collections import namedtuple
from functools import partial
from inspect import isfunction

from nemo.collections.asr.modules.transformer.transformer_modules import MultiHeadAttention, PositionWiseFF
from nemo.collections.asr.parts.submodules.adapters.attention_adapter_mixin import AttentionAdapterModuleMixin
from nemo.collections.asr.parts.utils import adapter_utils
from nemo.collections.common.parts import form_attention_mask
from nemo.core.classes.mixins import adapter_mixins

# structs

Memory = namedtuple('Memory', ['mem', 'compressed_mem'])
Return = namedtuple('Return', ['loss', 'aux_loss', 'is_last_batch'])


# helper functions

def to(t):
    return {'dtype': t.dtype, 'device': t.device}

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

def default(x, val):
    if x is not None:
        return x
    return val if not isfunction(val) else val()

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def reshape_dim(t, dim, split_dims):
    shape = list(t.shape)
    num_dims = len(shape)
    dim = (dim + num_dims) % num_dims
    shape[dim:dim+1] = split_dims
    return t.reshape(shape)

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

def queue_fifo(*args, length, dim=-2):
    queue = torch.cat(args, dim=dim)
    if length > 0:
        return split_at_index(dim, -length, queue)

    device = queue.device
    shape = list(queue.shape)
    shape[dim] = 0
    return queue, torch.empty(shape, device = device)

def shift(x):
    *_, i, j = x.shape
    zero_pad = torch.zeros((*_, i, i), **to(x))
    x = torch.cat([x, zero_pad], -1)
    l = i + j - 1
    x = x.view(*_, -1)
    zero_pad = torch.zeros(*_, -x.size(-1) % l, **to(x))
    shifted = torch.cat([x, zero_pad], -1).view(*_, -1, l)
    return shifted[..., :i, i - 1:]

def iterate_tensor(t):
    length = t.shape[0]
    for ind in range(length):
        yield t[ind]

# full attention for calculating auxiliary reconstruction loss

def full_attn(q, k, v, dropout_fn = None):
    *_, dim = q.shape
    dots = torch.einsum('bhid,bhjd->bhij', q, k) * (dim ** -0.5)
    attn = dots.softmax(dim=-1)
    if dropout_fn is not None:
        attn = dropout_fn(attn)
    return torch.einsum('bhij,bhjd->bhid', attn, v)

# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        out = cast_tuple(out)
        ret = (out[0] + x), *out[1:]
        return ret

class GRUGating(nn.Module):
    def __init__(self, dim, fn, mogrify = False):
        super().__init__()
        self.dim = dim
        self.fn = fn
        self.gru = nn.GRUCell(dim, dim)
        self.mogrify = Mogrifier(dim, factorize_k = dim // 4) if mogrify else None

    def forward(self, x, **kwargs):
        batch, dim = x.shape[0], self.dim
        out = self.fn(x, **kwargs)
        (y, *rest) = cast_tuple(out)

        if self.mogrify is not None:
            y, x = self.mogrify(y, x)

        gated_output = self.gru(
            y.reshape(-1, dim),
            x.reshape(-1, dim)
        )

        gated_output = gated_output.reshape(batch, -1, dim)
        ret = gated_output, *rest
        return ret

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)
    
# feedforward

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

class AvgCompress(nn.Module):
    def __init__(self, num_speakers=4):
        super().__init__()
        self.num_speakers = num_speakers

    def forward(self, mem):
        return mem
    
class ConvCompress(nn.Module):
    def __init__(self, dim, ratio = 4):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, ratio, stride = ratio)

    def forward(self, mem):
        mem = mem.transpose(1, 2)
        compressed_mem = self.conv(mem)
        return compressed_mem.transpose(1, 2)

class SqCompress(nn.Module):
    def __init__(self, num_speakers=4):
        super().__init__()
        self.num_speakers = num_speakers

    def forward(self, mem):
        pass

class CompressiveAttention(nn.Module):
    def __init__(self, dim, seq_len, mem_len, cmem_len, cmem_ratio = 4, heads = 8, attn_dropout = 0., dropout = 0., reconstruction_attn_dropout = 0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.heads = heads
        self.dim_head = dim // heads
        self.seq_len = seq_len
        self.mem_len = mem_len
        self.cmem_len = cmem_len
        self.cmem_ratio = cmem_ratio
        self.scale = self.dim_head ** (-0.5)

        self.compress_mem_fn = ConvCompress(dim, cmem_ratio)

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_kv = nn.Linear(dim, dim * 2, bias = False)
        self.to_out = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = nn.Dropout(dropout)

        self.reconstruction_attn_dropout = nn.Dropout(reconstruction_attn_dropout)

    def forward(self, x, memories = None, pos_emb = None, input_mask = None, calc_memory = True, **kwargs):
        b, t, e, h, dim_h = *x.shape, self.heads, self.dim_head
        
        memories = default(memories, (None, None))
        mem, cmem = memories

        init_empty_mem = lambda: torch.empty(b, 0, e, **to(x))
        mem = default(mem, init_empty_mem)
        cmem = default(cmem, init_empty_mem)

        mem_len = mem.shape[1]
        cmem_len = cmem.shape[1]

        q = self.to_q(x)

        kv_input = torch.cat((cmem, mem, x), dim=1)
        kv_len = kv_input.shape[1]
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        merge_heads = lambda x: reshape_dim(x, -1, (-1, dim_h)).transpose(1, 2)
        q, k, v = map(merge_heads, (q, k, v))

        k, v = map(lambda x: x.expand(-1, h, -1, -1), (k, v))

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = max_neg_value(dots)

        if pos_emb is not None:
            pos_emb = pos_emb[:, -kv_len:].type(q.dtype)
            pos_dots = torch.einsum('bhid,hjd->bhij', q, pos_emb) * self.scale
            pos_dots = shift(pos_dots)
            dots = dots + pos_dots

        if input_mask is not None:
            mask = input_mask[:, None, :, None] * input_mask[:, None, None, :]
            mask = F.pad(mask, (mem_len + cmem_len, 0), value = True)
            dots.masked_fill_(~mask, mask_value)

        total_mem_len = mem_len + cmem_len
        mask = torch.ones(t, t + total_mem_len, **to(x)).triu_(diagonal = 1 + total_mem_len).bool()
        dots.masked_fill_(mask[None, None, ...], mask_value)

        attn = dots.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(b, t, -1)
        logits = self.to_out(out)
        logits = self.dropout(logits)

        new_mem = mem
        new_cmem = cmem
        aux_loss = torch.zeros(1, requires_grad = True, **to(q))

        if self.seq_len > t or not calc_memory:
            return logits, Memory(new_mem, new_cmem), aux_loss

        # calculate memory and compressed memory

        old_mem, new_mem = queue_fifo(mem, x, length = self.mem_len, dim = 1)
        old_mem_padding = old_mem.shape[1] % self.cmem_ratio
        if old_mem_padding != 0:
            old_mem = F.pad(old_mem, (0, 0, old_mem_padding, 0), value = 0.)

        if old_mem.shape[1] == 0 or self.cmem_len <= 0:
            return logits, Memory(new_mem, new_cmem), aux_loss

        compressed_mem = self.compress_mem_fn(old_mem.detach())
        old_cmem, new_cmem = split_at_index(1, -self.cmem_len, torch.cat((cmem, compressed_mem), dim=1))

        if not self.training:
            return logits, Memory(new_mem, new_cmem), aux_loss

        # calculate compressed memory auxiliary loss if training

        self.to_kv.weight.detach_()

        cmem_k, cmem_v = self.to_kv(compressed_mem).chunk(2, dim=-1)
        cmem_k, cmem_v = map(merge_heads, (cmem_k, cmem_v))
        cmem_k, cmem_v = map(lambda x: x.expand(-1, h, -1, -1), (cmem_k, cmem_v))

        old_mem_range = slice(- min(mem_len, self.mem_len) - self.seq_len, -self.seq_len)
        old_mem_k, old_mem_v = map(lambda x: x[:, :, old_mem_range].clone(), (k, v))

        q, old_mem_k, old_mem_v = map(torch.detach, (q, old_mem_k, old_mem_v))

        attn_fn = partial(full_attn, dropout_fn = self.reconstruction_attn_dropout)

        aux_loss = F.mse_loss(
            attn_fn(q, old_mem_k, old_mem_v),
            attn_fn(q, cmem_k, cmem_v)
        )
        return logits, Memory(new_mem, new_cmem), aux_loss

class CompressiveTransformer(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        depth,
        emb_dim = None,
        memory_layers = None,
        enhanced_recurrence = True,
        mem_len = None,
        cmem_len = None,
        cmem_ratio = 4,
        heads = 8,
        gru_gated_residual = True,
        mogrify_gru = False,
        attn_dropout = 0.,
        ff_glu = False,
        ff_dropout = 0.,
        attn_layer_dropout = 0.,
        reconstruction_attn_dropout = 0.,
        reconstruction_loss_weight = 1.
    ):
        super().__init__()
        emb_dim = default(emb_dim, dim)
        mem_len = default(mem_len, seq_len)
        cmem_len = default(cmem_len, mem_len // cmem_ratio)
        memory_layers = default(memory_layers, list(range(1, depth + 1)))

        assert mem_len >= seq_len, 'length of memory should be at least the sequence length'
        assert cmem_len >= (mem_len // cmem_ratio), f'length of compressed memory should be at least the memory length divided by the compression ratio {int(mem_len // cmem_ratio)}'
        assert all([layer > 0 and layer <= depth for layer in memory_layers]), 'one of the indicated memory layers is invalid'

        self.seq_len = seq_len

        self.depth = depth
        self.memory_layers = list(memory_layers)
        self.enhanced_recurrence = enhanced_recurrence

        # self.token_emb = nn.Embedding(num_tokens, emb_dim)
        # self.to_model_dim = nn.Identity() if emb_dim == dim else nn.Linear(emb_dim, dim)

        seq_and_mem_len = seq_len + mem_len + cmem_len
        self.pos_emb = nn.Parameter(torch.zeros(heads, seq_and_mem_len, dim // heads))
        
        # self.to_logits = nn.Sequential(
        #     nn.Identity() if emb_dim == dim else nn.Linear(dim, emb_dim),
        #     nn.Linear(emb_dim, num_speakers)
        # )

        wrapper = partial(GRUGating, dim, mogrify = mogrify_gru) if gru_gated_residual else Residual

        self.attn_layers = nn.ModuleList([wrapper(PreNorm(dim, CompressiveAttention(dim, seq_len, mem_len, cmem_len, cmem_ratio, heads, dropout = attn_layer_dropout, attn_dropout = attn_dropout, reconstruction_attn_dropout = reconstruction_attn_dropout))) for _ in range(depth)])
        self.ff_layers = nn.ModuleList([wrapper(PreNorm(dim, FeedForward(dim, dropout = ff_dropout, glu = ff_glu))) for _ in range(depth)])

        self.reconstruction_loss_weight = reconstruction_loss_weight

    def forward(self, x, memories = None, mask = None):
        # x = self.token_emb(x)
        # x = self.to_model_dim(x)
        b, t, d = x.shape

        assert t <= self.seq_len, f'input contains a sequence length {t} that is greater than the designated maximum sequence length {self.seq_len}'

        memories = default(memories, (None, None))
        mem, cmem = memories

        num_memory_layers = len(self.memory_layers)
        init_empty_mem = lambda: torch.empty(num_memory_layers, b, 0, d, **to(x))
        mem = default(mem, init_empty_mem)
        cmem = default(cmem, init_empty_mem)

        total_len = mem.shape[2] + cmem.shape[2] + self.seq_len
        pos_emb = self.pos_emb[:, (self.seq_len - t):total_len]

        next_mem = []
        next_cmem = []
        aux_loss = torch.tensor(0., requires_grad = True, **to(x))

        if self.enhanced_recurrence:
            mem = torch.roll(mem, -1, 0)
            cmem = torch.roll(cmem, -1, 0)

        mem_iter, cmem_iter = map(iterate_tensor, (mem, cmem))

        for ind, (attn, ff) in enumerate(zip(self.attn_layers, self.ff_layers)):
            layer_num = ind + 1

            use_memory = layer_num in self.memory_layers
            memories = (next(mem_iter), next(cmem_iter)) if use_memory else None

            x, (mem_out, cmem_out), layer_aux_loss = attn(x, memories = memories, calc_memory = use_memory, input_mask = mask, pos_emb = pos_emb)
            x,  = ff(x)

            aux_loss = aux_loss + layer_aux_loss

            if not use_memory:
                continue

            next_mem.append(mem_out)
            next_cmem.append(cmem_out)

        # out = self.to_logits(x)
        out = x

        next_mem, next_cmem = map(torch.stack, (next_mem, next_cmem))
        next_mem, next_cmem = map(torch.detach, (next_mem, next_cmem))

        aux_loss = aux_loss * self.reconstruction_loss_weight / num_memory_layers
        return out, Memory(mem = next_mem, compressed_mem = next_cmem), aux_loss
        # return out, Memory(mem = next_mem, compressed_mem = next_cmem)
    

# class AutoregressiveWrapper(nn.Module):
#     def __init__(self, net, ignore_index = -100, pad_value = 0):
#         super().__init__()
#         self.pad_value = pad_value
#         self.ignore_index = ignore_index

#         self.net = net
#         self.seq_len = net.seq_len

#     @torch.no_grad()
#     def generate(self, start_tokens, seq_len, eos_token = None, temperature = 1., filter_logits_fn = top_k, filter_thres = 0.9, **kwargs):
#         was_training = self.net.training
#         num_dims = len(start_tokens.shape)

#         if num_dims == 1:
#             start_tokens = start_tokens[None, :]

#         b, t = start_tokens.shape

#         self.net.eval()

#         out = start_tokens

#         # take care of default masking

#         full_mask_like = lambda x: torch.full_like(x, True, dtype=torch.bool, device=x.device)

#         mask = kwargs.pop('mask', None)
#         if mask is None:
#             mask = full_mask_like(out)

#         # take care of a primed sequence of any length

#         mem = None
#         *primes, out = out.split(self.seq_len, dim=1)
#         *prime_masks, mask = mask.split(self.seq_len, dim=1)

#         for prime, prime_mask in zip(primes, prime_masks):
#             _, mem, _ = self.net(prime, memories = mem, mask = prime_mask, **kwargs)

#         # generate until hit sequence length

#         input_len = out.shape[1]

#         for _ in range(seq_len):
#             logits, mem, aux_loss = self.net(out[:, -input_len:], memories = mem, mask = mask[:, -input_len:], **kwargs)
#             logits = logits[:, -1, :]
#             filtered_logits = filter_logits_fn(logits, thres = filter_thres)
#             probs = F.softmax(filtered_logits / temperature, dim=-1)
#             sample = torch.multinomial(probs, 1)

#             # unlike most models, inputs start from sequence length of 1 once full sequence length is filled

#             out = torch.cat((out, sample), dim=-1)
#             mask = F.pad(mask, (0, 1), value=True)

#             # append sample to accumulated output            

#             input_len = input_len % self.seq_len
#             input_len += 1

#             if eos_token is not None and (sample == eos_token).all():
#                 break

#         out = out[:, t:]

#         if num_dims == 1:
#             out = out.squeeze(0)

#         self.net.train(was_training)
#         return out

#     def forward(self, x, max_batch_size = None, return_loss = False, **kwargs):
#         pad = partial(pad_sequence, batch_first = True, padding_value = self.pad_value)

#         if not return_loss:
#             if not isinstance(x, torch.Tensor):
#                 x = pad(x)
#             return self.net(x, **kwargs)

#         if isinstance(x, torch.Tensor):
#             xi = x[:, :-1]
#             xo = x[:, 1:]
#         else:
#             xi = pad(list(map(lambda t: t[:-1], x)))
#             xo = pad(list(map(lambda t: t[1:], x)))

#         # help auto-solve an area of confusion around input masks in auto-regressive
#         # if user supplies a mask that is only off by one from the source sequence, resolve it for them
#         mask = kwargs.pop('mask', None)
#         if mask is not None and mask.shape[1] == x.shape[1]:
#             mask = mask[:, :-1]

#         segment_fn = lambda x: x.split(self.seq_len, dim=1)
#         (xi, xo) = map(segment_fn, (xi, xo))

#         num_segments = len(xi)
#         mask = segment_fn(mask) if mask is not None else ((None,) * num_segments)

#         max_batch_size = x.shape[0] if max_batch_size is None else max_batch_size
#         split_batch_fn = lambda x: x.split(max_batch_size, dim=0)

#         grad_accumulate_every = math.ceil(x.shape[0] / max_batch_size)
#         mems = [None] * grad_accumulate_every

#         for xi_seg, xo_seg, mask_seg in zip(xi, xo, mask):
#             xi_seg, xo_seg = map(split_batch_fn, (xi_seg, xo_seg))
#             mask_seg = split_batch_fn(mask_seg) if mask_seg is not None else ((None,) * grad_accumulate_every)

#             new_mems = []
#             for ind, (xi_seg_b, xo_seg_b, mask_seg_b, mem) in enumerate(zip(xi_seg, xo_seg, mask_seg, mems)):
#                 is_last = ind == (grad_accumulate_every - 1)

#                 logits, new_mem, aux_loss = self.net(xi_seg_b, mask = mask_seg_b, memories = mem, **kwargs)
#                 new_mems.append(new_mem)

#                 loss = F.cross_entropy(logits.transpose(1, 2), xo_seg_b, ignore_index = self.ignore_index)
#                 yield Return(loss, aux_loss, is_last)

#             mems = new_mems