import torch
from torch import nn
from transformers import AutoTokenizer
from pathlib import Path

from src.causal_model import CausalLM
from src.utils import get_state_dict_convert_fun, get_model_args

def from_pretrained(
        model_dir: Path,
        strict=True,
    ) -> tuple[CausalLM, str]:
        assert model_dir.is_dir()

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True
        )

        def _load_model(model_dir: Path):
            if model_dir is None:
                return None, None
            model_name = model_dir.name
            model_args = get_model_args(model_name)
            if model_args.n_vocab != len(tokenizer):
                print(
                    f"WARNING: {model_name}: model_args.n_vocab ({model_args.n_vocab}) != len(tokenizer) "
                    f"({len(tokenizer)})"
                )
            model = CausalLM.from_pretrained(
                model_dir,
                model_args,
                strict,
                get_state_dict_convert_fun(model_name),
            )
            model.eval()
            return model, model_name

        model, model_name = _load_model(model_dir)
        return model, model_name



if __name__ == "__main__":
    model_name = "Qwen1.5-0.5B-Chat"
    model_dir = Path(f"checkpoints/{model_name}")
    model, model_name = from_pretrained(model_dir, strict=True)
    print(f"Loaded model: {model_name}")
    print(f"Model args: {model.args}")
    
    print("Model parameter names:")
    for name, _ in model.named_parameters():
        print(name)