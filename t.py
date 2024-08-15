import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import math
from transformers import set_seed, logging

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

logging.set_verbosity_error()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    # Ensure all paths are absolute
    model_path = os.path.abspath(os.path.expanduser(args.model_path))
    question_file_path = os.path.abspath(os.path.expanduser(args.question_file))
    answers_file = os.path.abspath(os.path.expanduser(args.answers_file))
    image_folder = os.path.abspath(os.path.expanduser(args.image_folder))

    print(f"Model path: {model_path}")
    print(f"Question file path: {question_file_path}")
    print(f"Image folder: {image_folder}")
    print(f"Answer file: {answers_file}")
    
    set_seed(0)
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    if not os.path.exists(question_file_path):
        raise FileNotFoundError(f"Question file not found: {question_file_path}")

    questions = [json.loads(q) for q in open(question_file_path, "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Tokenize the prompt with the image token
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        if input_ids is None or len(input_ids) == 0:
            print(f"Error: Input IDs are not generated for question {idx}.")
            continue

        image_path = os.path.join(image_folder, image_file)
        if not os.path.exists(image_path):
            print(f"Image {image_path} does not exist for question {idx}.")
            continue

        # Process image
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)
        if len(image_tensor) > 0:
            image_tensor = image_tensor[0]
        else:
            print(f"Failed to process image: {image_path} for question {idx}.")
            continue

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        try:
            output_ids = model.generate(
                inputs=input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True
            )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({
                "question_id": idx,
                "prompt": cur_prompt,
                "text": outputs,
                "answer_id": ans_id,
                "model_id": model_name,
                "metadata": {}
            }) + "\n")
            ans_file.flush()
        except Exception as e:
            error_message = (
                f"Error during model.generate for question {idx}: {e}\n"
                f"Image tensor shape: {image_tensor.shape}\n"
                f"Input IDs shape: {input_ids.shape}\n"
                f"Question ID: {idx}\n"
                f"Image Path: {image_path}\n"
                f"Prompt: {cur_prompt}"
            )
            print(error_message)
            ans_file.write(f"{error_message}\n")
            raise RuntimeError(error_message) from e

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
