import os
import random
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from retro_pytorch import RETRO, RETRODataset
from retro_pytorch.optimizer import get_optimizer
import numpy as np
from tqdm import tqdm
from configs import *

# train parameters
batch_size = 12
lr = 3e-4
epochs = 20
step_size = 5
gamma = 0.8
seed = 42

# checkpoint
CHECKPOINT = ''
#CHECKPOINT = 'output/retro_s512_b12_e31_0.494477.pt.weights'
total_epochs = 0


# random seed settings
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

# instantiate dataset class
# which constructs the sequence and neighbors from memmapped chunk and neighbor information

train_ds = RETRODataset(
    num_sequences = NUM_SEQS,
    num_chunks = NUM_CHUNKS,
    num_neighbors = NUM_NEIGHBORS,
    chunk_size = CHUNK_SIZE,
    seq_len = SEQ_LEN,
    chunk_memmap_path = f'{TRAIN_DATA_PATH}/train.chunks.dat',
    chunk_nn_memmap_path = f'{TRAIN_DATA_PATH}/train.chunks.knn.dat',
    seq_memmap_path = f'{TRAIN_DATA_PATH}/train.seq.dat'
)

train_dl = DataLoader(train_ds, batch_size = batch_size)


# one forwards and backwards

retro = RETRO(
    max_seq_len = SEQ_LEN,                      # max sequence length
    enc_dim = 896,                           # encoder model dimension
    enc_depth = 3,                           # encoder depth
    dec_dim = 768,                           # decoder model dimensions
    dec_depth = 12,                          # decoder depth
    dec_cross_attn_layers = (1, 3, 6, 9),    # decoder cross attention layers (with causal chunk cross attention)
    heads = 8,                               # attention heads
    dim_head = 64,                           # dimension per head
    dec_attn_dropout = 0.25,                 # decoder attention dropout
    dec_ff_dropout = 0.25                    # decoder feedforward dropout
).cuda()

parameters = filter(lambda p: p.requires_grad, retro.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM\n' % parameters)

# 载入 checkpoint
if os.path.exists(CHECKPOINT):
    checkpoint = torch.load(CHECKPOINT)
    retro.load_state_dict(checkpoint['model_state_dict'])
    total_epochs = checkpoint['epoch']
    last_loss = checkpoint['loss']
    print(f"Loaded {CHECKPOINT}: epochs= {total_epochs}, loss= {last_loss:.6f}\n")
    # 调整起始lr
    lr = lr * gamma**(total_epochs//step_size)


## optimizer
optimizer = get_optimizer(retro.parameters(), lr = lr, wd = 0.01)

## lr stchedule
lr_scheduler = StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)


best_loss = 1e+4

for epoch in range(epochs):
    epoch_loss = 0

    print(f'Epoch: {epoch+1}/{epochs}, lr: {lr_scheduler.get_last_lr()[0]:.2e}')

    pbar = tqdm(train_dl)
    for seq, retrieved in pbar:
        seq, retrieved = seq.cuda(), retrieved.cuda()

        loss = retro(
            seq,
            retrieved,
            return_loss = True
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss / len(train_dl)

        pbar.set_description(f'loss: {loss:.6f}')
        pbar.refresh()

    pbar.close()

    if epoch>0 and epoch_loss <= best_loss: # 第1轮作为warmup
        best_loss = epoch_loss
        # 保存
        torch.save({
                    'epoch'                : total_epochs+epoch+1,
                    'model_state_dict'     : retro.state_dict(),
                    #'optimizer_state_dict' : optimizer.state_dict(),
                    'loss'                 : epoch_loss,
        }, f"./output/retro_s{SEQ_LEN}_b{batch_size}_e{total_epochs+epoch+1}_{epoch_loss:.6f}.pt.weights")

    print(f"loss : {epoch_loss:.6f} best_loss : {best_loss:.6f}\n")
    lr_scheduler.step()

'''
# 清除训练时缓存， 用在notebook时

retro = None
torch.cuda.empty_cache()

import gc
gc.collect()
'''