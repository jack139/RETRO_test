import torch
import torch.nn.functional as F
from functools import partial
from retro_pytorch import RETRO
from retro_pytorch.training import top_k, top_p, exists, gumbel_sample, safe_cat, knn_chunks_from_seq_chunks
from retro_pytorch.retrieval import SOS_ID, EOS_ID, faiss_read_index, tokenize, get_tokenizer
from einops import rearrange
from configs import *

# checkpoint
CHECKPOINT = 'output/retro_s512_b12_e34_0.400526.pt.weights'
total_epochs = 0


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

# 载入 checkpoint
checkpoint = torch.load(CHECKPOINT)
retro.load_state_dict(checkpoint['model_state_dict'])
total_epochs = checkpoint['epoch']
last_loss = checkpoint['loss']
print(f"Loaded {CHECKPOINT}: epochs= {total_epochs}, loss= {last_loss:.6f}\n")


# 生成token
def generate(start = None, retrieved = None, filter_fn = top_k, filter_thres = 0.9, temperature = 1.0):
    assert filter_fn in {top_k, top_p}, 'filter function must be either top-k or nucleus'

    faiss_index = faiss_read_index(f'{TRAIN_DATA_PATH}/tmp/.index/knn.index')

    fetch_knn_chunks_fn = partial(
        knn_chunks_from_seq_chunks,
        knn = 2,
        chunk_size = CHUNK_SIZE,
        num_chunks = NUM_CHUNKS,
        chunks_memmap_path = f'{TRAIN_DATA_PATH}/train.chunks.dat',
        faiss_index = faiss_index
    )

    device = next(retro.parameters()).device

    # if not prime tokens given, assume sampling from SOS token with batch size of 1

    if not exists(start):
        start = torch.full((1, 1), SOS_ID, device = device).long()

    b, start_seq_len = start.shape

    # move onto same device as RETRO

    start = start.to(device)

    # prepare retrieval related variables

    if start_seq_len >= CHUNK_SIZE:
        seq_index = (start_seq_len // CHUNK_SIZE) * CHUNK_SIZE
        past_seq_chunks = rearrange(start[:, :seq_index], 'b (n c) -> (b n) c', c = CHUNK_SIZE)

        retrieved = fetch_knn_chunks_fn(past_seq_chunks)
        retrieved = rearrange(retrieved, '(b n) k c -> b n k c', b = b)

    # get starting sequence index

    out = start

    # sampling loop

    for i in range(start_seq_len - 1, SEQ_LEN):

        logits = retro(out, retrieved = retrieved)
        logits = logits[:, i]

        logits = filter_fn(logits, thres = filter_thres)
        sampled = gumbel_sample(logits, temperature = temperature, dim = -1)
        sampled = rearrange(sampled, 'b -> b 1')

        out = torch.cat((out, sampled), dim = 1)

        # early terminate if all EOS

        is_eos_tokens = (out == EOS_ID)

        if is_eos_tokens.any(dim = -1).all():

            # mask out everything after the eos tokens

            shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
            mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
            out = out.masked_fill(mask, retro.pad_id)
            break

        # when the sequence length is a multiple of the chunk size
        # retrieve the next set of knns

        curr_seq_len = out.shape[-1]

        if (curr_seq_len % CHUNK_SIZE) == 0:
            last_chunk = rearrange(out, 'b (c n) -> b c n', n = CHUNK_SIZE)[:, -1]

            knn_chunks = fetch_knn_chunks_fn(last_chunk)

            # concat retrieved knn chunks to all retrieved
            # to be sent to Retro for chunked cross attention at the next iteration

            knn_chunks = rearrange(knn_chunks, 'b k r -> b 1 k r')
            retrieved = safe_cat(retrieved, knn_chunks, dim = 1)

            print(f'retrieved at {curr_seq_len} / {SEQ_LEN}')

    return out

if __name__ == '__main__':

    prompt_text = '始皇东游'

    # 生成 prompt token
    prompt = tokenize(prompt_text, add_special_tokens=False)
    print(prompt)

    # 生成文本
    out = generate(prompt)

    # token转为为文字
    tokenizer = get_tokenizer()
    txt=tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(txt[0].replace(' ',''))
