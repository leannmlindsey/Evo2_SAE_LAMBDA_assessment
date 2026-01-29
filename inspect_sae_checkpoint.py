#!/usr/bin/env python3
"""
Inspect Evo2 SAE Checkpoint
===========================
Run this first to understand the SAE checkpoint structure.

Usage:
    python inspect_sae_checkpoint.py
"""

import os
import torch
from pathlib import Path
from huggingface_hub import list_repo_files, hf_hub_download

def inspect_sae():
    repo_id = "Goodfire/Evo-2-Layer-26-Mixed"
    sae_dir = Path("/home/lindseylm/evo2/sae_weights")

    print(f"Looking for SAE weights in: {sae_dir}")
    
    print("=" * 60)
    print("Inspecting Evo2 SAE Checkpoint")
    print("=" * 60)
    
    # List files in repo
    print(f"\n1. Files in HuggingFace repo '{repo_id}':")
    print("-" * 40)
    files = list_repo_files(repo_id)
    for f in files:
        print(f"   {f}")
    
    # Find checkpoint files
    print(f"\n2. Local files in {sae_dir}:")
    print("-" * 40)
    
    checkpoint_files = []
    if sae_dir.exists():
        for f in sae_dir.rglob("*"):
            if f.is_file():
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"   {f.relative_to(sae_dir)}: {size_mb:.1f} MB")
                if f.suffix in ['.pt', '.pth', '.bin', '.safetensors']:
                    checkpoint_files.append(f)
    else:
        print(f"   Directory not found! Run setup script first.")
        return
    
    # Inspect checkpoint structure
    print(f"\n3. Checkpoint structure:")
    print("-" * 40)
    
    for ckpt_file in checkpoint_files:
        print(f"\n   File: {ckpt_file.name}")
        
        try:
            if ckpt_file.suffix == '.safetensors':
                from safetensors import safe_open
                with safe_open(ckpt_file, framework="pt") as f:
                    keys = list(f.keys())
                    print(f"   Format: safetensors")
                    print(f"   Keys ({len(keys)}):")
                    for k in keys[:20]:  # First 20 keys
                        tensor = f.get_tensor(k)
                        print(f"      {k}: {tensor.shape} ({tensor.dtype})")
                    if len(keys) > 20:
                        print(f"      ... and {len(keys) - 20} more")
            else:
                ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
                print(f"   Format: PyTorch checkpoint")
                print(f"   Type: {type(ckpt)}")
                
                if isinstance(ckpt, dict):
                    print(f"   Top-level keys: {list(ckpt.keys())}")
                    
                    for key in ckpt.keys():
                        val = ckpt[key]
                        if isinstance(val, torch.Tensor):
                            print(f"      {key}: Tensor {val.shape} ({val.dtype})")
                        elif isinstance(val, dict):
                            print(f"      {key}: dict with {len(val)} keys")
                            # Show first few subkeys
                            for i, (subkey, subval) in enumerate(val.items()):
                                if i >= 5:
                                    print(f"         ... and {len(val) - 5} more")
                                    break
                                if isinstance(subval, torch.Tensor):
                                    print(f"         {subkey}: {subval.shape}")
                                else:
                                    print(f"         {subkey}: {type(subval)}")
                        else:
                            print(f"      {key}: {type(val)}")
                            
                elif isinstance(ckpt, torch.nn.Module):
                    print(f"   Model architecture:")
                    print(ckpt)
                    
                else:
                    print(f"   Content: {ckpt}")
                    
        except Exception as e:
            print(f"   Error loading: {e}")
    
    # Also check for config files
    print(f"\n4. Config files:")
    print("-" * 40)
    for config_file in sae_dir.rglob("*.json"):
        print(f"\n   {config_file.name}:")
        import json
        with open(config_file) as f:
            config = json.load(f)
        for k, v in config.items():
            print(f"      {k}: {v}")
    
    for config_file in sae_dir.rglob("*.yaml"):
        print(f"\n   {config_file.name}:")
        with open(config_file) as f:
            print(f.read())

    # Inspect Evo2 model structure to find correct hook points
    print(f"\n5. Evo2 Model Hook Points (layer 26 area):")
    print("-" * 40)
    
    try:
        from evo2 import Evo2
        print("   Loading Evo2 7B model (this may take a minute)...")
        model = Evo2('evo2_7b')
        
        print("\n   Available layer names containing '26':")
        for name, module in model.model.named_modules():
            if '26' in name:
                print(f"      {name}: {type(module).__name__}")
        
        print("\n   Full model structure (blocks 24-28):")
        for name, module in model.model.named_modules():
            # Only show blocks near 26
            if any(f'blocks.{i}' in name for i in [24, 25, 26, 27, 28]):
                if name.count('.') <= 3:  # Don't go too deep
                    print(f"      {name}: {type(module).__name__}")
                    
    except Exception as e:
        print(f"   Could not load Evo2 model: {e}")
        print("   (This is expected if you haven't run the full test yet)")

    print("\n" + "=" * 60)
    print("Inspection complete!")
    print("=" * 60)
    print("\nUse this information to update the SAE loading code in")
    print("run_prophage_detection.py")


if __name__ == "__main__":
    inspect_sae()
