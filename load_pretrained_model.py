# run_model_vqa.py
import sys
from transformers import AutoTokenizer, AutoConfig
from llava.model.builder import LlavaModel, AutoImageProcessor

def load_pretrained_model(model_path, model_base, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = AutoConfig.from_pretrained(model_path)
    model = LlavaModel.from_pretrained(model_path, config=config)
    image_processor = AutoImageProcessor.from_pretrained(model_path)
    context_len = config.max_position_embeddings

    return tokenizer, model, image_processor, context_len

# Importing the main function from model_vqa
from model_vqa import eval_model, parse_args

if __name__ == "__main__":
    args = parse_args()
    eval_model(args)
