#!/usr/bin/env python3
"""Test local NVIDIA embeddings."""

import os
import sys

from dotenv import load_dotenv
load_dotenv()

import torch

def test_nvidia_embeddings():
    """Test loading NVIDIA Nemotron Embed locally."""
    print(f"=== GPU Info ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"\n=== Loading nvidia/llama-nemotron-embed-1b-v2 ===")
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(
            "nvidia/llama-nemotron-embed-1b-v2",
            device="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
        )
        
        # Test embedding
        texts = [
            "Buy signal for AAPL based on momentum indicators",
            "Risk management stop-loss at 2% below entry",
            "Small cap stocks with high volume breakout patterns"
        ]
        
        embeddings = model.encode(texts)
        
        print(f"✓ Model loaded successfully!")
        print(f"✓ Embedding shape: {embeddings.shape}")
        print(f"✓ VRAM used: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        # Test similarity
        from sentence_transformers import util
        sim = util.cos_sim(embeddings[0], embeddings[1])
        print(f"✓ Similarity (trade signal ↔ risk mgmt): {sim.item():.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("NVIDIA Local Embeddings Test")
    print("=" * 60)
    
    success = test_nvidia_embeddings()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Local NVIDIA embeddings ready!")
        print("  Model: nvidia/llama-nemotron-embed-1b-v2")
        print("  Dimension: 2048")
        print("  VRAM: ~5 GB")
    else:
        print("✗ Test failed. Check errors above.")
    print("=" * 60)
