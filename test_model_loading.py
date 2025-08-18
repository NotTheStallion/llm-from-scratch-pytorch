import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

from src.causal_model import CausalLM
from src.model_args import ModelArgs
from main import load_model
from tqdm import tqdm





def main_test():
    model_name = "Qwen3-0.6B"
    model_dir = Path(f"checkpoints/{model_name}")
    model_dir = Path("checkpoints/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
    model, model_name, model_args = load_model(model_name, model_dir, strict=True)
    
    # Load the model using Hugging Face
    hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype="auto", trust_remote_code=True, cache_dir="checkpoints")
    
    # for name, param in model.named_parameters():
    #     if "lm_head.weight" in name:
    #         print(f"Model parameter: {name}, shape: {param.shape}")
    
    # for name, hf_param in hf_model.named_parameters():
    #     if "lm_head.weight" in name:
    #         print(f"Hugging Face model parameter: {name}, shape: {hf_param.shape}")
    
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
    
    # for name in hf_model_params.keys():
    #     if name not in model_params:
    #         print(f"Parameter '{name}' not found in custom model.")
    
    
    
    
    # # Checking values
    # print("Comparing parameter values...")
    # for name, param in model.named_parameters():
    #     if "lm_head.weight" not in name:
    #         hf_param = hf_model.get_parameter(name)
    #         if not torch.allclose(param.float(), hf_param.float(), atol=1e-200, rtol=1e-200):
    #             print(f"Parameter '{name}' values do not match between custom model and Hugging Face model.")
    #     else:
    #         lm_head_weight = hf_model.get_parameter("model.embed_tokens.weight")
    #         if not torch.allclose(param.float(), lm_head_weight.float(), atol=1e-200, rtol=1e-200):
    #             print(f"Parameter '{name}' values do not match between custom model and Hugging Face model.")


def test_inference():
    model_name = "Qwen3-0.6B"
    model_dir = Path(f"checkpoints/{model_name}")
    model_dir = Path("checkpoints/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
    model, model_name, model_args = load_model(model_name, model_dir, strict=True)
    tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B", trust_remote_code=True
        )
    
    # Load the model using Hugging Face
    hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype="float32", device_map="cpu", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(hf_model)

    prompt = "Where is china ?"
    
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cpu")
    
    print(input_ids)
    hf_logits = hf_model.forward(input_ids).logits
    print(f"Hugging Face logits shape: {hf_logits.shape}")
    
    logits = model.forward(input_ids)
    print(f"Custom model logits shape: {logits.shape}")
    
    # Take the last logits vector and apply softmax
    hf_probs = torch.softmax(hf_logits[0, -1], dim=-1)
    custom_probs = torch.softmax(logits[0, -1], dim=-1)
    
    print(hf_probs[:10])  # Print the first 10 probabilities from Hugging Face model
    print(custom_probs[:10])  # Print the first 10 probabilities from custom model
    
    # Compute the average difference, min, and max in probabilities for all logits
    hf_probs_all = torch.softmax(hf_logits, dim=-1)
    custom_probs_all = torch.softmax(logits, dim=-1)
    
    diff = torch.abs(hf_probs_all - custom_probs_all)
    avg_diff = torch.mean(diff).item()
    min_diff = torch.min(diff).item()
    max_diff = torch.max(diff).item()
    
    print(f"Average difference in probabilities: {avg_diff}")
    print(f"Minimum difference in probabilities: {min_diff}")
    print(f"Maximum difference in probabilities: {max_diff}")
    
    
    


if __name__ == "__main__":
    main_test()
    # test_inference()