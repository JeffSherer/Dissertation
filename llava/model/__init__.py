try:
    from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
    from .language_model.llava_mpt import LlavaMptForCausalLM, LlavaMptConfig
    from .language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig
    
    from transformers import AutoConfig, AutoModelForCausalLM
    
    # Registering configurations and models
    AutoConfig.register("llava_llama", LlavaConfig)
    AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
    
    AutoConfig.register("llava_mpt", LlavaMptConfig)
    AutoModelForCausalLM.register(LlavaMptConfig, LlavaMptForCausalLM)
    
    AutoConfig.register("llava_mistral", LlavaMistralConfig)
    AutoModelForCausalLM.register(LlavaMistralConfig, LlavaMistralForCausalLM)
    
    print("Models and configurations registered successfully.")
    
except ImportError as e:
    print(f"Failed to import: {e}")
    raise
