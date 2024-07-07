import torch
from flash_attn.flash_attn_interface import flash_attn_func
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Flash Attention and Transformers import successful!")
