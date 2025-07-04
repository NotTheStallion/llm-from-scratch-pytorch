import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

from src.causal_model import CausalLM
from src.model_args import ModelArgs
from main import from_pretrained
from tqdm import tqdm





def main_test():
    model_name = "Qwen1.5-0.5B-Chat"
    model_dir = Path(f"checkpoints/{model_name}")
    model, model_name, model_args = from_pretrained(model_dir, strict=True)
    
    # Load the model using Hugging Face
    hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", torch_dtype="auto", device_map="cpu", trust_remote_code=True, cache_dir="/beegfs/mkherraz")

    for name, param in model.named_parameters():
        if "lm_head.weight" in name:
            print(f"Model parameter: {name}, shape: {param.shape}")
    
    for name, hf_param in hf_model.named_parameters():
        if "lm_head.weight" in name:
            print(f"Hugging Face model parameter: {name}, shape: {hf_param.shape}")
    
    # Compare the names and shapes of parameters
    model_params = {name: param.shape for name, param in model.named_parameters()}
    hf_model_params = {name: hf_param.shape for name, hf_param in hf_model.named_parameters()}
    
    for name, shape in model_params.items():
        if name in hf_model_params:
            if shape != hf_model_params[name]:
                print(f"Parameter '{name}' shape mismatch: {shape} vs {hf_model_params[name]}")
        elif "lm_head.weight" in name:
            print(f"Parameter '{name}' not found in Hugging Face model, but it is a lm_head.weight. Copy of {hf_model_params['model.embed_tokens.weight']} will be used.")
        else:
            print(f"Parameter '{name}' not found in Hugging Face model.")
    
    for name in hf_model_params.keys():
        if name not in model_params:
            print(f"Parameter '{name}' not found in custom model.")
    
    
    
    
    # Checking values
    print("Comparing parameter values...")
    for name, param in model.named_parameters():
        if "lm_head.weight" not in name:
            hf_param = hf_model.get_parameter(name)
            if not torch.allclose(param.float(), hf_param.float(), atol=1e-200, rtol=1e-200):
                print(f"Parameter '{name}' values do not match between custom model and Hugging Face model.")
        else:
            lm_head_weight = hf_model.get_parameter("model.embed_tokens.weight")
            if not torch.allclose(param.float(), lm_head_weight.float(), atol=1e-200, rtol=1e-200):
                print(f"Parameter '{name}' values do not match between custom model and Hugging Face model.")
    


if __name__ == "__main__":
    main_test()