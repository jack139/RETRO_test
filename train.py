import torch
from retro_pytorch import RETRO
from transformers import AutoTokenizer, EncoderDecoderModel

column = "whole_func_string"
encoder_name = "microsoft/codebert-base"
decoder_name = "gpt2"
encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_name)
decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_name)
decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

