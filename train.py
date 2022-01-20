import torch
from retro_pytorch import RETRO
from transformers import AutoTokenizer, EncoderDecoderModel

column = "whole_func_string"
encoder_name = "microsoft/codebert-base"
decoder_name = "gpt2"
encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_name)
decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_name)
decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

# 中文的句子embedding考虑 "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
retro_ds = RetroDataset(
    "code_search_net",
    "flax-sentence-embeddings/st-codesearch-distilroberta-base",
    encoder_tokenizer,
    decoder_tokenizer,
    dataset_config="python",
    column=column,
    batch_size=4,
    k=2,
    n_perc=2
)
