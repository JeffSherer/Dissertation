import json
import os

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def compare_configs(config1, config2):
    keys_to_compare = [
        'vocab_size', 
        'model_type', 
        'tokenizer_padding_side', 
        'tokenizer_model_max_length',
        'num_attention_heads',
        'num_hidden_layers',
        'hidden_size'
    ]
    for key in keys_to_compare:
        if config1.get(key) != config2.get(key):
            print(f"Config mismatch on {key}: {config1.get(key)} != {config2.get(key)}")
            return False
    return True

def check_compatibility(bbf_path, mistral_path):
    bbf_config = load_json(os.path.join(bbf_path, 'config.json'))
    mistral_config = load_json(os.path.join(mistral_path, 'config.json'))

    # Compare model configurations
    configs_compatible = compare_configs(bbf_config, mistral_config)
    if not configs_compatible:
        print("Configurations are not compatible.")
        return False

    # Check if tokenizer files exist in both directories
    tokenizer_files = ['tokenizer_config.json', 'special_tokens_map.json', 'tokenizer.model']
    for file_name in tokenizer_files:
        mistral_file_path = os.path.join(mistral_path, file_name)
        if not os.path.exists(mistral_file_path):
            print(f"{file_name} does not exist in {mistral_path}.")
            return False

    print("Configurations and tokenizer files are compatible.")
    return True

if __name__ == "__main__":
    bbf_path = "/users/jjls2000/sharedscratch/Dissertation/results/BBF-NoMod"
    mistral_path = "/users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-med-v1.5-mistral-7b"
    
    compatible = check_compatibility(bbf_path, mistral_path)
    if compatible:
        print("The models are compatible. You can proceed with copying files if needed.")
    else:
        print("The models are not compatible. Do not copy the files.")
