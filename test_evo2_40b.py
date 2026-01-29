#!/usr/bin/env python3
"""
Test if Evo2 40B model can be loaded on H200s.
"""

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")

print("\nAttempting to load evo2_40b...")
print("(This may take several minutes and use multiple GPUs)")

try:
    from evo2 import Evo2

    # Try loading 40B model
    model = Evo2("evo2_40b")

    print(f"\n✓ Successfully loaded evo2_40b!")
    print(f"  Device: {next(model.model.parameters()).device}")

    # Check d_hidden - it might be different for 40B
    # Try to find the hidden dimension
    for name, module in model.model.named_modules():
        if hasattr(module, 'weight') and len(module.weight.shape) == 2:
            print(f"  Layer {name}: weight shape {module.weight.shape}")
            break

    # Test with a short sequence
    print("\nTesting with short sequence...")
    test_seq = "ATCGATCGATCGATCGATCG" * 10  # 200bp
    toks = model.tokenizer.tokenize(test_seq)
    toks = torch.tensor(toks, dtype=torch.long).unsqueeze(0).cuda()

    with torch.no_grad():
        output = model.model(toks)

    print(f"  ✓ Forward pass successful!")
    print(f"  Output shape: {output[0].shape if isinstance(output, tuple) else output.shape}")

except Exception as e:
    print(f"\n✗ Failed to load evo2_40b: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
