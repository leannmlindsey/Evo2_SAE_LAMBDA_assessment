#!/usr/bin/env python3
"""
Check Evo2 tokenizer vocabulary size and token mappings.
"""

from evo2 import Evo2

print("Loading Evo2 model (just for tokenizer)...")
model = Evo2("evo2_7b")

tokenizer = model.tokenizer

print("\n" + "=" * 60)
print("Evo2 Tokenizer Info")
print("=" * 60)

print(f"\nTokenizer type: {type(tokenizer)}")
print(f"vocab_size: {tokenizer.vocab_size}")

# Try tokenizing each base
print("\n" + "=" * 60)
print("Token IDs for DNA bases and sequences")
print("=" * 60)

test_seqs = ['A', 'T', 'C', 'G', 'N', 'a', 't', 'c', 'g', 'ATCG', 'atcg', 'NNNN', 'ATCGATCG']
for seq in test_seqs:
    try:
        tokens = tokenizer.tokenize(seq)
        print(f"  '{seq}' -> {tokens}")
    except Exception as e:
        print(f"  '{seq}' -> ERROR: {e}")

# Check what ASCII characters map to what tokens
print("\n" + "=" * 60)
print("ASCII character token mappings (printable chars)")
print("=" * 60)

for i in range(32, 127):
    char = chr(i)
    try:
        tokens = tokenizer.tokenize(char)
        print(f"  '{char}' (ASCII {i}) -> {tokens}")
    except Exception as e:
        print(f"  '{char}' (ASCII {i}) -> ERROR: {e}")

# Check safe attributes only
print("\n" + "=" * 60)
print("Safe tokenizer attributes")
print("=" * 60)

safe_attrs = ['vocab_size', 'pad_token_id', 'eos_token_id', 'bos_token_id',
              'unk_token_id', 'mask_token_id', 'cls_token_id', 'sep_token_id']
for attr in safe_attrs:
    if hasattr(tokenizer, attr):
        try:
            val = getattr(tokenizer, attr)
            print(f"  {attr}: {val}")
        except:
            pass

print("\nDone!")
