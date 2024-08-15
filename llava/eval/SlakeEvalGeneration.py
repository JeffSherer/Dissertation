import json
from transformers import AutoTokenizer, CLIPImageProcessor
from PIL import Image
import torch
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
import os

# Path to the JSON file
json_file_path = '/users/jjls2000/sharedscratch/Dissertation/Slake1.0/augmented_filtered/BBF_train_medium.json'

# Load the JSON file
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Extract one sample
sample = data[0]  # Assuming data is a list of samples
print("Sample:", sample)

# Load the tokenizer and image processor
tokenizer_path = "/users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-med-v1.5-mistral-7b-BBF-3"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

# Extract the text and image from the sample
conversations = sample['conversations']
text = conversations[0]['value']  # Assuming the first conversation entry contains the relevant text
image_path = sample['image']  # Adjust this based on your sample structure

# Tokenize the text
input_ids = tokenizer(text, return_tensors="pt").input_ids

# Process the image
image_path = os.path.join('/users/jjls2000/sharedscratch/Dissertation/data/imgs-1', image_path)
image = Image.open(image_path).convert("RGB")
image_tensor = image_processor(images=image, return_tensors="pt").pixel_values

# Move tensors to the correct device
input_ids = input_ids.to('cuda')
image_tensor = image_tensor.to('cuda')

# Load the model
model_path = "/users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-med-v1.5-mistral-7b-BBF-3"
model = LlavaMistralForCausalLM.from_pretrained(model_path).to('cuda')

# Check if the vision tower attribute is set
if not hasattr(model.get_model(), 'vision_tower'):
    print("Vision tower attribute missing, setting it up.")
    model.get_model().vision_tower = model.get_model().get_vision_tower()

# Ensure the vision tower is an instance of CLIPVisionTower
if not isinstance(model.get_model().vision_tower, CLIPVisionTower):
    raise ValueError("Expected vision_tower to be an instance of CLIPVisionTower")

# Generate the output
output = model.generate(
    inputs=input_ids,
    images=image_tensor,
    num_beams=5,  # Adjust parameters as needed
    max_length=50,  # Adjust max length as needed
)

# Decode the output to text
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Output:", output_text)
