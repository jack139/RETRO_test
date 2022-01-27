import torch
from torch.utils.data import DataLoader
from retro_pytorch import RETRO, RETRODataset
import numpy as np



# mock data constants
NUM_CHUNKS = 66 # news1.txt NUM_CHUNKS = 66 NUM_DOCS = 1    NUM_SEQS = 3
NUM_SEQS = 3
CHUNK_SIZE = 64
NUM_NEIGHBORS = 2

# instantiate dataset class
# which constructs the sequence and neighbors from memmapped chunk and neighbor information

train_ds = RETRODataset(
    num_sequences = NUM_SEQS,
    num_chunks = NUM_CHUNKS,
    num_neighbors = NUM_NEIGHBORS,
    chunk_size = CHUNK_SIZE,
    seq_len = 2048,
    chunk_memmap_path = './test_data/train.chunks.dat',
    chunk_nn_memmap_path = './test_data/train.chunks.knn.dat',
    seq_memmap_path = './test_data/train.seq.dat'
)

train_dl = iter(DataLoader(train_ds, batch_size = 2))

# one forwards and backwards

retro = RETRO(
    max_seq_len = 2048,                      # max sequence length
    enc_dim = 896,                           # encoder model dimension
    enc_depth = 3,                           # encoder depth
    dec_dim = 768,                           # decoder model dimensions
    dec_depth = 12,                          # decoder depth
    dec_cross_attn_layers = (1, 3, 6, 9),    # decoder cross attention layers (with causal chunk cross attention)
    heads = 8,                               # attention heads
    dim_head = 64,                           # dimension per head
    dec_attn_dropout = 0.25,                 # decoder attention dropout
    dec_ff_dropout = 0.25                    # decoder feedforward dropout
)#.cuda()

'''
seq, retrieved = map(lambda t: t.cuda(), next(train_dl))

# seq       - (2, 2049)         - 1 extra token since split by seq[:, :-1], seq[:, 1:]
# retrieved - (2, 32, 2, 128)   - 128 since chunk + continuation, each 64 tokens

loss = retro(
    seq,
    retrieved,
    return_loss = True
)

loss.backward()
'''