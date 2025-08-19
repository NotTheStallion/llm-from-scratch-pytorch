import torch

from pathlib import Path
from collections import OrderedDict
import safetensors.torch
from typing import Callable, List, Tuple

from model_args import ModelArgs


def get_device(device="auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            dc = torch.cuda.device_count()
            device = f"cuda:{dc-1}"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        return torch.device(device)
    try:
        return torch.device(device)
    except:
        raise ValueError(f"Invalid device: {device}")


def get_model_state_dict_filenames(model_dir: Path) -> List[Path]:
    assert model_dir.is_dir()
    if len(list(model_dir.glob("*.safetensors"))) > 0:
        return list(model_dir.glob("*.safetensors"))
    if len(list(model_dir.glob("*.pth"))) > 0:
        return list(model_dir.glob("*.pth"))
    if len(list(model_dir.glob("*.bin"))) > 0:
        return list(model_dir.glob("*.bin"))
    raise ValueError(f"Cannot find model files in {model_dir}")


def _load_model_state_dict(fn: Path | List[Path]) -> OrderedDict:
    state_dict = OrderedDict()
    if not isinstance(fn, (list, tuple)):
        fn = [fn]
    for f in fn:
        if f.suffix == ".safetensors":
            state_dict.update(safetensors.torch.load_file(f, device="cpu"))
        elif f.suffix in [".pth", ".bin"]:
            state_dict.update(torch.load(f, map_location="cpu"))
        else:
            raise ValueError(f"Unknown file type: {f.suffix}")
    return state_dict


def load_model_state_dict(model_path: Path | List[Path]) -> OrderedDict:
    try:  # model_path is a directory
        model_fns = get_model_state_dict_filenames(model_path)
        return _load_model_state_dict(model_fns)
    except:
        return _load_model_state_dict(model_path)


def print_model_state_dict(model_dir: Path, save_fn: Path = None) -> str:
    state_dict = load_model_state_dict(model_dir)
    state_dict_info = ""
    for k, v in state_dict.items():
        state_dict_info += f"{k}: {[*v.shape]}\n"
    if save_fn is not None:
        with open(save_fn, "w") as f:
            f.write(state_dict_info)
    return state_dict_info


MODEL_ARGS_MAP = {
    "TinyLlama-1.1B-Chat-v1.0": ModelArgs(
        llm_type="llama",
        n_vocab=32000,
        dim=2048,
        n_layers=22,
        n_heads=32,
        n_kv_heads=4,
        ffn_hidden_dim=5632,
        max_batch_size=1,
        max_seq_len=2048,
    ),
    "LLama2-7b": ModelArgs(
        llm_type="llama",
        n_vocab=32000,
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=None,
        ffn_hidden_dim=11008,
        max_batch_size=1,
        max_seq_len=4096,
    ),
    "Qwen1.5-0.5B-Chat": ModelArgs(
        llm_type="qwen",
        n_vocab=151936,
        dim=1024,
        n_layers=24,
        n_heads=16,
        n_kv_heads=16,
        ffn_hidden_dim=2816,
        norm_eps=1e-6,
        rope_theta=1000000.0,
        max_batch_size=1,
        max_seq_len=4096,  # 32768,
    ),
    "Qwen1.5-1.8B-Chat": ModelArgs(
        llm_type="qwen",
        n_vocab=151936,
        dim=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=None,
        ffn_hidden_dim=5504,
        norm_eps=1e-6,
        rope_theta=1000000.0,
        max_batch_size=1,
        max_seq_len=4096,  # 32768,
    ),
    "Qwen3-0.6B": ModelArgs(
        llm_type="qwen",
        n_vocab=151_936,
        dim=1024,
        n_layers=28,
        d_head=128,
        n_heads=16,
        n_kv_heads=8,
        qk_rms_norm=True,
        ffn_hidden_dim=3072,
        norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_batch_size=2,
        max_seq_len=40_960,  # 32768,
        dtype=torch.bfloat16,
    )
}

def get_model_args(model_name: str) -> ModelArgs:
    print(model_name)
    return MODEL_ARGS_MAP.get(model_name, None)

