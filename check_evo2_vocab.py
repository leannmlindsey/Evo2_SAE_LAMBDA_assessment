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

# Check common attributes
print(f"\nTokenizer type: {type(tokenizer)}")

# Try to get vocab size
if hasattr(tokenizer, 'vocab_size'):
    print(f"vocab_size attribute: {tokenizer.vocab_size}")

if hasattr(tokenizer, '__len__'):
    print(f"len(tokenizer): {len(tokenizer)}")

if hasattr(tokenizer, 'get_vocab'):
    vocab = tokenizer.get_vocab()
    print(f"get_vocab() size: {len(vocab)}")
    print(f"Vocabulary: {vocab}")

if hasattr(tokenizer, 'vocab'):
    print(f"vocab attribute: {tokenizer.vocab}")

# Try tokenizing each base
print("\n" + "=" * 60)
print("Token IDs for DNA bases")
print("=" * 60)

test_seqs = ['A', 'T', 'C', 'G', 'N', 'ATCG', 'NNNN', 'ATCGATCG']
for seq in test_seqs:
    tokens = tokenizer.tokenize(seq)
    print(f"  '{seq}' -> {tokens}")

# Check all attributes
print("\n" + "=" * 60)
print("All tokenizer attributes")
print("=" * 60)
for attr in dir(tokenizer):
    if not attr.startswith('_'):
        try:
            val = getattr(tokenizer, attr)
            if not callable(val):
                print(f"  {attr}: {val}")
        except:
            pass

print("\nDone!")
