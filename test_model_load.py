from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

model_path = "/users/jjls2000/sharedscratch/Dissertation/checkpoints/llava-med-7b-slake-delta"
try:
    model = LlavaLlamaForCausalLM.from_pretrained(model_path)
    print(model)
except Exception as e:
    print(f"Error loading the model: {e}")
