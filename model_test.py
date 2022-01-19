import torch
from retro_pytorch import RETRO

retro = RETRO(
    num_tokens = 20000,                      # number of tokens
    chunk_size = 64,                         # the chunk size that is indexed and retrieved (needed for proper relative positions as well as causal chunked cross attention)
    max_seq_len = 2048,                      # max sequence length
    enc_dim = 896,                           # encoder model dim
    enc_depth = 2,                           # encoder depth
    dec_dim = 796,                           # decoder model dim
    dec_depth = 12,                          # decoder depth
    dec_cross_attn_layers = (3, 6, 9, 12),   # decoder cross attention layers (with causal chunk cross attention)
    heads = 8,                               # attention heads
    dim_head = 64,                           # dimension per head
    dec_attn_dropout = 0.25,                 # decoder attention dropout
    dec_ff_dropout = 0.25                    # decoder feedforward dropout
)

seq = torch.randint(0, 20000, (2, 2048 + 1))      # plus one since it is split into input and labels for training
retrieved = torch.randint(0, 20000, (2, 32, 2, 128)) # retrieved tokens - (batch, num chunks, num retrieved neighbors, retrieved chunk with continuation)

loss = retro(seq, retrieved, return_loss = True)
loss.backward()
