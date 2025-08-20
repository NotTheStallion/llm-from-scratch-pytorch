import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

from src.causal_model import CausalLM
from src.model_args import ModelArgs, get_dtype
from main import load_model
from tqdm import tqdm
from test_modules import Qwen3Model




def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None):

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(token_ids)[:, -1]
            next_token = torch.argmax(out, dim=-1, keepdim=True)

            if (eos_token_id is not None
                   and torch.all(next_token == eos_token_id)):
               break

            yield next_token
            
            token_ids = torch.cat([token_ids, next_token], dim=1)





def load_weights_into_qwen(model, param_config, params):
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(right.clone().detach() if isinstance(right, torch.Tensor) else torch.tensor(right))

    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    for l in range(param_config["n_layers"]):
        block = model.trf_blocks[l]
        att = block.att

        # Q, K, V projections
        att.W_query.weight = assign(
            att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        att.W_key.weight = assign(
            att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        att.W_value.weight = assign(
            att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )

        # Output projection
        att.out_proj.weight = assign(
            att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )

        # QK norms
        if hasattr(att, "q_norm") and att.q_norm is not None:
            att.q_norm.scale = assign(
                att.q_norm.scale,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight"
            )
        if hasattr(att, "k_norm") and att.k_norm is not None:
            att.k_norm.scale = assign(
                att.k_norm.scale,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight"
            )

        # Attention layernorm
        block.norm1.scale = assign(
            block.norm1.scale,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # Feedforward weights
        block.ff.fc1.weight = assign(
            block.ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        block.ff.fc2.weight = assign(
            block.ff.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        block.ff.fc3.weight = assign(
            block.ff.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )
        block.norm2.scale = assign(
            block.norm2.scale,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # Final normalization and output head
    model.final_norm.scale = assign(model.final_norm.scale, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params:
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        # Model uses weight tying, hence we reuse the embedding layer weights here
        print("Model uses weight tying.")
        model.out_head.weight = assign(model.out_head.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")




def load_custom_and_gt_model():
    import json
    import os
    from pathlib import Path
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download, snapshot_download

    QWEN3_CONFIG = {
        "vocab_size": 151_936,           # Vocabulary size
        "context_length": 40_960,        # Context length that was used to train the model
        "emb_dim": 1024,                 # Embedding dimension
        "n_heads": 16,                   # Number of attention heads
        "n_layers": 28,                  # Number of layers
        "hidden_dim": 3072,              # Size of the intermediate dimension in FeedForward
        "head_dim": 128,                 # Size of the heads in GQA
        "qk_norm": True,                 # Whether to normalize queries and values in GQA
        "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
        "rope_base": 1_000_000.0,        # The base in RoPE's "theta"
        "dtype": torch.float32,         # Lower-precision dtype to reduce memory usage
    }
    
    torch.manual_seed(123)
    model = Qwen3Model(QWEN3_CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B", trust_remote_code=True
        )
    
    print(model(torch.tensor([1, 2, 3]).unsqueeze(0)))

    repo_id = "Qwen/Qwen3-0.6B"

    local_dir = "checkpoints"
    
    weights_file = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
        local_dir=local_dir,
    )
    weights_dict = load_file(weights_file)

    load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
    model.to("cpu")
    
    prompt = "Give me a short introduction to large language models."

    input_token_ids = tokenizer.encode(prompt)
    

    input_token_ids_tensor = torch.tensor(input_token_ids, device="cpu").unsqueeze(0)


    for token in generate_text_basic_stream(
        model=model,
        token_ids=input_token_ids_tensor,
        max_new_tokens=500,
        eos_token_id=tokenizer.eos_token_id
    ):
        token_id = token.squeeze(0).tolist()
        print(
            tokenizer.decode(token_id),
            end="",
            flush=True
        )


    del weights_dict





def main_test():
    import json
    import os
    from pathlib import Path
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download, snapshot_download

    QWEN3_CONFIG = {
        "vocab_size": 151_936,           # Vocabulary size
        "context_length": 40_960,        # Context length that was used to train the model
        "emb_dim": 1024,                 # Embedding dimension
        "n_heads": 16,                   # Number of attention heads
        "n_layers": 28,                  # Number of layers
        "hidden_dim": 3072,              # Size of the intermediate dimension in FeedForward
        "head_dim": 128,                 # Size of the heads in GQA
        "qk_norm": True,                 # Whether to normalize queries and values in GQA
        "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
        "rope_base": 1_000_000.0,        # The base in RoPE's "theta"
        "dtype": torch.float32,         # Lower-precision dtype to reduce memory usage
    }
    
    
    
    
    model_name = "Qwen3-0.6B"
    repo_id = "Qwen/Qwen3-0.6B"
    local_dir = "checkpoints"
    weights_file = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
        local_dir=local_dir,
    )
    
    print(f"Loading model from {weights_file}")
    
    
    # model_dir = Path(f"checkpoints/{model_name}")
    model_dir = Path(weights_file).parent
    model, model_name, model_args = load_model(model_name, model_dir, strict=True)
    tokenizer = AutoTokenizer.from_pretrained(
            repo_id, trust_remote_code=True
        )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the GT model
    gt_model = Qwen3Model(QWEN3_CONFIG)

    weights_dict = load_file(weights_file)

    load_weights_into_qwen(gt_model, QWEN3_CONFIG, weights_dict)
    gt_model.to("cpu")
    
    # for name, param in model.named_parameters():
    #     if "lm_head.weight" in name:
    #         print(f"Model parameter: {name}, shape: {param.shape}")
    
    # for name, hf_param in hf_model.named_parameters():
    #     if "lm_head.weight" in name:
    #         print(f"Hugging Face model parameter: {name}, shape: {hf_param.shape}")

    
    # Compare the names and shapes of parameters
    model_params = {name: param.shape for name, param in model.named_parameters()}
    gt_model_params = {name: hf_param.shape for name, hf_param in gt_model.named_parameters()}

    param_match = {"trf_blocks": "model.layers", "att": "self_attn", "scale": "weight", "ff": "mlp",
                   "W_query": "q_proj", "W_key": "k_proj", "W_value": "v_proj", "out_proj": "o_proj", "q_norm": "q_norm", "k_norm": "k_norm",
                   "fc1": "gate_proj", "fc2": "up_proj", "fc3": "down_proj", "norm1": "input_layernorm", "norm2": "post_attention_layernorm",
                   "final_norm": ".norm", "out_head": "lm_head", "tok_emb": "embed_tokens"}
    
    for name, shape in gt_model_params.items():
        # Modify parameter names using param_match dictionary
        modified_name = name
        for custom_key, hf_key in param_match.items():
            modified_name = modified_name.replace(custom_key, hf_key)

        # print(modified_name, shape)

        found = False
        for param_name in model_params:
            if modified_name in param_name:
                assert model_params[param_name] == shape, f"Shape mismatch for {modified_name}: {model_params[param_name]} != {shape}"
                found = True
                
        
        if not found:
            raise ValueError(f"Parameter '{modified_name}' not found in custom model parameters.")
    

    # for name, shape in model_params.items():
    #     print(name, shape)
    
    print("Comparing parameter names and shapes ... OK")
    
    
    # Checking values
    for name, shape in gt_model_params.items():
        # Modify parameter names using param_match dictionary
        modified_name = name
        for custom_key, hf_key in param_match.items():
            modified_name = modified_name.replace(custom_key, hf_key)

        for param_name, param_value in model.named_parameters():
            if modified_name in param_name:
                assert torch.allclose(param_value, gt_model.state_dict()[name], atol=1e-5), \
                    f"Value mismatch for {modified_name}"
    

    print("Comparing parameter values ... OK")
    
    
    

def test_inference():
    import json
    import os
    from pathlib import Path
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download, snapshot_download

    QWEN3_CONFIG = {
        "vocab_size": 151_936,           # Vocabulary size
        "context_length": 40_960,        # Context length that was used to train the model
        "emb_dim": 1024,                 # Embedding dimension
        "n_heads": 16,                   # Number of attention heads
        "n_layers": 28,                  # Number of layers
        "hidden_dim": 3072,              # Size of the intermediate dimension in FeedForward
        "head_dim": 128,                 # Size of the heads in GQA
        "qk_norm": True,                 # Whether to normalize queries and values in GQA
        "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
        "rope_base": 1_000_000.0,        # The base in RoPE's "theta"
        "dtype": torch.float32,         # Lower-precision dtype to reduce memory usage
    }
    
    
    
    
    model_name = "Qwen3-0.6B"
    repo_id = "Qwen/Qwen3-0.6B"
    local_dir = "checkpoints"
    weights_file = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
        local_dir=local_dir,
    )
    
    print(f"Loading model from {weights_file}")
    
    
    # model_dir = Path(f"checkpoints/{model_name}")
    model_dir = Path(weights_file).parent
    model, model_name, model_args = load_model(model_name, model_dir, strict=True)
    tokenizer = AutoTokenizer.from_pretrained(
            repo_id, trust_remote_code=True
        )
    tokenizer.pad_token = tokenizer.eos_token

    print(model)
    
    # Load the GT model
    gt_model = Qwen3Model(QWEN3_CONFIG)

    weights_dict = load_file(weights_file)

    load_weights_into_qwen(gt_model, QWEN3_CONFIG, weights_dict)
    gt_model.to("cpu")
    
    print(gt_model)

    prompt = "Where is china ?"
    
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cpu")
    
    print(input_ids)
    
    num_tokens = input_ids.shape[1]
    input_ids = torch.randint(0, model_args.n_vocab, (model_args.max_batch_size, num_tokens))  # (batch_size, num_tokens)
    
    # print(gt_model.forward(input_ids))
    gt_logits = gt_model.forward(input_ids)
    print(f"Hugging Face logits shape: {gt_logits.shape}")
    
    logits = model.forward(input_ids)
    print(f"Custom model logits shape: {logits.shape}")
    
    # Compare logits from both models
    assert gt_logits.shape == logits.shape, "Logits shape mismatch between models"

    logits_diff = torch.abs(gt_logits - logits)
    avg_logits_diff = torch.mean(logits_diff).item()
    min_logits_diff = torch.min(logits_diff).item()
    max_logits_diff = torch.max(logits_diff).item()

    print(f"Average difference in logits: {avg_logits_diff}")
    print(f"Minimum difference in logits: {min_logits_diff}")
    print(f"Maximum difference in logits: {max_logits_diff}")
    
    
    # Take the last logits vector and apply softmax
    gt_probs = torch.softmax(gt_logits[0, -1], dim=-1)
    custom_probs = torch.softmax(logits[0, -1], dim=-1)
    
    print(gt_probs[:10])  # Print the first 10 probabilities from Hugging Face model
    print(custom_probs[:10])  # Print the first 10 probabilities from custom model
    
    # Compute the average difference, min, and max in probabilities for all logits
    gt_probs_all = torch.softmax(gt_logits, dim=-1)
    custom_probs_all = torch.softmax(logits, dim=-1)
    
    diff = torch.abs(gt_probs_all - custom_probs_all)
    avg_diff = torch.mean(diff).item()
    min_diff = torch.min(diff).item()
    max_diff = torch.max(diff).item()
    
    print(f"Average difference in probabilities: {avg_diff}")
    print(f"Minimum difference in probabilities: {min_diff}")
    print(f"Maximum difference in probabilities: {max_diff}")
    
    
    


if __name__ == "__main__":
    main_test()
    test_inference()
    # load_custom_and_gt_model()