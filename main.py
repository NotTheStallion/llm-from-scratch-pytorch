import torch
from torch import nn
from transformers import AutoTokenizer
from pathlib import Path

from src.causal_model import CausalLM
from src.utils import get_model_args
from src.model_args import ModelArgs

def load_model(
        model_dir: Path,
        strict=True,
    ) -> tuple[CausalLM, str, ModelArgs]:
        assert model_dir.is_dir()

        if model_dir is None:
            return None, None, None
        model_name = model_dir.name
        model_args = get_model_args(model_name)
        
        model = CausalLM.from_pretrained(
            model_dir,
            model_args,
            strict
        )
        model.eval()
        return model, model_name, model_args



if __name__ == "__main__":
    model_name = "Qwen1.5-0.5B-Chat"
    model_dir = Path(f"checkpoints/{model_name}")
    model, model_name, model_args = load_model(model_dir, strict=True)
    tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True
        )

    if model_args.n_vocab != len(tokenizer):
                print(
                    f"WARNING: {model_name}: model_args.n_vocab ({model_args.n_vocab}) != len(tokenizer) "
                    f"({len(tokenizer)})"
                )
    
    print(f"Loaded model: {model_name}")
    print(f"Model args: {model.args}")
    
    # print("Model parameter names:")
    # for name, _ in model.named_parameters():
    #     print(name)
    
    chat_template = tokenizer.chat_template
    # print(f"Chat template: {chat_template}")
    
    prompt = "What is a dog ?"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    prompt = tokenizer.apply_chat_template(messages)
    
    print("="*20)
    print(f"Prompt: {tokenizer.decode(prompt)}")
    
    # inputs = tokenizer(prompt, return_tensors="pt")
    # input_ids = inputs["input_ids"]
    # print(f"Input IDs shape: {input_ids.shape}")
    
    num_generations = 1  # Number of generations
    max_length = 50  # Maximum length of generated text
    
    
    tokenizer.bos_token_id = 151644
    print(f"bos token id: {tokenizer.bos_token_id}")
    print(f"eos token id: {tokenizer.eos_token_id}")
    
    # print(f"Input IDs: {input_ids}")

    with torch.no_grad():
        for i in range(num_generations):
            # Manually implement generation using the model's forward method
            generated_ids = torch.tensor(prompt.copy()).unsqueeze(0)  # Add batch dimension [1, L]
            for _ in range(max_length):
                print(generated_ids)
                outputs = model.forward(generated_ids)
                print(f"Outputs shape: {outputs.shape}")  # [B, L, n_vocab]
                next_token_logits = outputs[:, -1, :]  # Get logits for the last token
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                print(f"Next token: {next_token}")  # [B, 1]
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                print(f"Generated IDs shape: {generated_ids.shape}")  # [B, L]

                # Stop generation if the end-of-sequence token is produced
                if next_token.item() == tokenizer.eos_token_id:
                    break
            
            generated_text = tokenizer.decode(generated_ids[0])
            print(f"Generated text {i + 1}: {generated_text}")