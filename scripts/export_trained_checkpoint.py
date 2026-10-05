#!/usr/bin/env python3
"""Export a trained DFlash checkpoint into a standalone HuggingFace/vLLM compatible draft model directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer


def export_checkpoint(
    checkpoint_path: str | Path,
    target_model_path: str | Path,
    output_dir: str | Path,
    *,
    block_size: int = 16,
    mask_token_id: int = 151669,
    num_draft_layers: int = 5,
    target_layer_ids: list[int] | None = None,
    dtype: str = "bfloat16",
) -> Path:
    cp = Path(checkpoint_path).resolve()
    target = Path(target_model_path).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f">>> Reading target model config from: {target}")
    target_config = AutoConfig.from_pretrained(str(target))
    target_num_layers = int(getattr(target_config, "num_hidden_layers", 28))

    # Determine state dict location
    state_dict_path = None
    if (cp / "draft_state_dict.pt").is_file():
        state_dict_path = cp / "draft_state_dict.pt"
    elif (cp / "draft_export" / "draft_state_dict.pt").is_file():
        state_dict_path = cp / "draft_export" / "draft_state_dict.pt"
    elif cp.is_file() and cp.suffix == ".pt":
        state_dict_path = cp
    else:
        # Search inside subdirectories
        candidates = list(cp.glob("**/draft_state_dict.pt"))
        if candidates:
            state_dict_path = candidates[0]

    if state_dict_path is None or not state_dict_path.is_file():
        raise FileNotFoundError(f"Cannot find draft_state_dict.pt in {cp}")

    print(f">>> Loading draft weights from: {state_dict_path}")
    raw_state_dict = torch.load(str(state_dict_path), map_location="cpu", weights_only=True)
    
    # Strip prefixes if any
    clean_state_dict = {}
    for k, v in raw_state_dict.items():
        clean_k = k
        for prefix in ("draft_model.", "model.", "module."):
            if clean_k.startswith(prefix):
                clean_k = clean_k[len(prefix):]
        clean_state_dict[clean_k] = v

    # Resolve target layer ids
    if target_layer_ids is None:
        step = target_num_layers / num_draft_layers
        target_layer_ids = [int(round(i * step)) for i in range(num_draft_layers)]

    # Build DFlash Config JSON
    config_dict = target_config.to_dict()
    config_dict.update({
        "architectures": ["DFlashDraftModel"],
        "num_hidden_layers": num_draft_layers,
        "num_target_layers": target_num_layers,
        "block_size": block_size,
        "layer_types": ["full_attention"] * num_draft_layers,
        "torch_dtype": dtype,
        "target_model_name_or_path": str(target),
        "target_layer_ids": target_layer_ids,
        "dflash_config": {
            "target_layer_ids": target_layer_ids,
            "mask_token_id": mask_token_id,
            "block_size": block_size,
        },
    })

    config_file = out / "config.json"
    print(f">>> Writing draft model config to: {config_file}")
    with config_file.open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # Save weights as safetensors (preferred) and pytorch_model.bin
    try:
        from safetensors.torch import save_file
        safetensors_file = out / "model.safetensors"
        print(f">>> Saving safetensors weights to: {safetensors_file}")
        save_file(clean_state_dict, str(safetensors_file))
    except ImportError:
        pass

    bin_file = out / "pytorch_model.bin"
    print(f">>> Saving PyTorch bin weights to: {bin_file}")
    torch.save(clean_state_dict, str(bin_file))

    # Copy tokenizer files from target
    for file_name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.json",
        "vocab.json",
        "merges.txt",
    ):
        src_file = target / file_name
        if src_file.is_file():
            shutil.copy2(src_file, out / file_name)

    print(f"✅ Exported draft model successfully to: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained DFlash checkpoint to HuggingFace/vLLM format")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory (containing draft_state_dict.pt)")
    parser.add_argument("--target-model", required=True, help="Path to base/target model (e.g. Qwen3-4B)")
    parser.add_argument("--output-dir", required=True, help="Output directory to save exported draft model")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-draft-layers", type=int, default=5)
    parser.add_argument("--mask-token-id", type=int, default=151669)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    export_checkpoint(
        checkpoint_path=args.checkpoint,
        target_model_path=args.target_model,
        output_dir=args.output_dir,
        block_size=args.block_size,
        num_draft_layers=args.num_draft_layers,
        mask_token_id=args.mask_token_id,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
