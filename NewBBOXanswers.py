import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import re
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

def extract_bboxes(output_text):
    """
    Extracts bounding box coordinates from the output text.
    Assumes coordinates are given in the format (x1, y1, x2, y2).
    """
    bbox_pattern = r'\(\d+,\s*\d+,\s*\d+,\s*\d+\)'  # This is for (x1, y1, x2, y2)
    bboxes = re.findall(bbox_pattern, output_text)
    parsed_bboxes = []
    
    for bbox in bboxes:
        coords = list(map(int, re.findall(r'\d+', bbox)))
        parsed_bboxes.append(coords)
    
    return parsed_bboxes

def eval_model(args):
    # Ensure all paths are absolute
    model_path = os.path.abspath(os.path.expanduser(args.model_path))
    question_file_path = os.path.abspath(os.path.expanduser(args.question_file))
    answers_file = os.path.abspath(os.path.expanduser(args.answers_file))
    image_folder = os.path.abspath(os.path.expanduser(args.image_folder))

    set_seed(0)
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    if not os.path.exists(question_file_path):
        raise FileNotFoundError(f"Question file not found: {question_file_path}")

    questions = [json.loads(q) for q in open(question_file_path, "r")]
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"].replace(DEFAULT_IMAGE_TOKEN, '').strip()

        # Modify the prompt to explicitly ask for bounding box coordinates
        qs += " Please provide precise bounding box coordinates for the relevant objects."

        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = process_images([image], image_processor, model.config)[0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            # Log the model output
            print(f"Model output for question {idx}: {outputs}")

            # Extract bounding boxes from the output text
            bboxes = extract_bboxes(outputs)

            # Check if bounding boxes were found
            if not bboxes:
                print(f"No bounding boxes found for question {idx}. Model output: {outputs}")
            else:
                print(f"Bounding boxes for question {idx}: {bboxes}")

            # Write only the question_id and bbox to the output file
            ans_file.write(json.dumps({
                "question_id": idx,
                "bbox": bboxes
            }) + "\n")
            ans_file.flush()

        except Exception as e:
            error_message = (
                f"Error during model.generate for question {idx}: {e}\n"
                f"Image tensor shape: {image_tensor.shape}\n"
                f"Input IDs shape: {input_ids.shape}\n"
                f"Question ID: {idx}\n"
                f"Image Path: {os.path.join(args.image_folder, image_file)}\n"
                f"Prompt: {cur_prompt}"
            )
            print(error_message)
            ans_file.write(f"{error_message}\n")
            raise RuntimeError(error_message) from e

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/path/to/model")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/path/to/images")
    parser.add_argument("--question-file", type=str, default="/users/jjls2000/sharedscratch/Dissertation/Slake1.0/test_questions_fixed.jsonl")
    parser.add_argument("--answers-file", type=str, default="/users/jjls2000/sharedscratch/Dissertation/results/BBL-3-fixed_answers.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
