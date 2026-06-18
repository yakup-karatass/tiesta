#!/usr/bin/env python3
# ==============================================================================
# SPA-Engine — AI Router (PyTorch + C++ Bridge + Hugging Face)
# ==============================================================================
# This module implements the neural SPA Router: a lightweight PyTorch network
# that predicts which transformer layers to compute vs. bypass for each input
# token during LLM inference.
#
# Integration:
#   We use the Qwen/Qwen2.5-Coder-0.5B embedding layer to generate real 
#   semantic token embeddings for input text. These real embeddings are
#   then fed into our SPARouter.
#
# Architecture:
#   Input (Qwen token embedding, dim=896)
#     → Linear(896, 64) → ReLU
#     → Linear(64, total_layers) → Sigmoid
#     → Soft probabilities p ∈ [0,1]^total_layers
#     → Threshold at 0.5 → Binary mask {0, 1}^total_layers
#
# Usage:
#   python3 src/ai_router.py
# ==============================================================================

import sys
import os
import time
from typing import List, Set

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# -----------------------------------------------------------------------------
# Import the C++ spa_engine module.
# -----------------------------------------------------------------------------
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build")
sys.path.insert(0, build_dir)

import spa_engine  # noqa: E402


# ==============================================================================
# ANSI Escape Codes for Terminal Formatting
# ==============================================================================
class Style:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"
    RED     = "\033[31m"


def divider(width: int = 70) -> str:
    return "  " + "─" * width


def mask_to_str(probs: torch.Tensor, critical: Set[int], threshold: float = 0.5) -> str:
    chars = []
    for i in range(len(probs)):
        is_active = probs[i].item() >= threshold
        is_crit = i in critical
        if is_active and is_crit:
            chars.append(f"{Style.CYAN}█{Style.RESET}")
        elif is_active:
            chars.append(f"{Style.GREEN}█{Style.RESET}")
        elif is_crit:
            chars.append(f"{Style.RED}░{Style.RESET}")
        else:
            chars.append(f"{Style.DIM}░{Style.RESET}")
    return "[" + "".join(chars) + "]"


# ==============================================================================
# SPARouter — PyTorch Neural Network
# ==============================================================================
class SPARouter(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        total_layers: int = 40,
        hidden_dim: int = 64,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.total_layers = total_layers
        self.threshold = threshold

        self.network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, total_layers),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = self.forward(x)
            return (probs >= self.threshold).float()


def spa_loss(probs: torch.Tensor, critical_layers: Set[int], lam: float = 0.1, eps: float = 1e-7) -> tuple:
    latency_loss = probs.sum()
    critical_indices = torch.tensor(list(critical_layers), dtype=torch.long)
    critical_probs = probs[critical_indices]
    accuracy_loss = -torch.log(critical_probs.clamp(min=eps)).sum()
    total_loss = lam * latency_loss + accuracy_loss
    return total_loss, latency_loss, accuracy_loss


def train_step(router, optimizer, embedding, critical_layers, lam=0.1):
    optimizer.zero_grad()
    probs = router(embedding)
    total_loss, lat_loss, acc_loss = spa_loss(probs, critical_layers, lam)
    total_loss.backward()
    optimizer.step()
    return total_loss.item(), lat_loss.item(), acc_loss.item()


# ==============================================================================
# Main: Hugging Face Integration + Training + C++ Inference
# ==============================================================================
def main():
    # --- Configuration --------------------------------------------------------
    MODEL_NAME     = "Qwen/Qwen2.5-Coder-0.5B"
    TOTAL_LAYERS   = 40
    TRAIN_STEPS    = 100
    LEARNING_RATE  = 0.01
    LAMBDA         = 0.1
    WEIGHT_FILE    = "dummy_layer.bin"
    CRITICAL_LAYERS: Set[int] = {0, 1, 38, 39}

    torch.manual_seed(1337)

    # --- Header ---------------------------------------------------------------
    print()
    print(divider())
    print(f"{Style.BOLD}{Style.CYAN}    SPA-Engine{Style.RESET}"
          f"{Style.DIM}  AI Router + Hugging Face Integration{Style.RESET}")
    print(f"{Style.DIM}    Model: {MODEL_NAME}{Style.RESET}")
    print(divider())
    print()

    # =========================================================================
    # Phase 0: Load HF Model
    # =========================================================================
    print(f"{Style.BOLD}{Style.WHITE}  PHASE 0{Style.RESET}{Style.DIM}  Loading Transformer Embeddings{Style.RESET}")
    print(f"  {Style.DIM}Downloading/loading {MODEL_NAME} (this may take a moment)...{Style.RESET}")
    
    # We only need the embedding layer, so we can just load the base model.
    # To save memory, we can use trust_remote_code if needed.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    hf_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Get embedding dimension
    EMBED_DIM = hf_model.config.hidden_size
    print(f"  {Style.GREEN}✓{Style.RESET} Model loaded. Embedding Dimension: {EMBED_DIM}")
    print()

    def get_token_embeddings(text: str) -> torch.Tensor:
        """Tokenize text and return sequence of embeddings."""
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            # Get the input embeddings directly
            embeddings = hf_model.get_input_embeddings()(tokens.input_ids).squeeze(0)
        # Cast to float32 to match SPARouter weights (Qwen is often bfloat16)
        return embeddings.to(torch.float32), tokens.input_ids.squeeze(0)

    # --- Initialize Router + Optimizer ----------------------------------------
    router = SPARouter(embed_dim=EMBED_DIM, total_layers=TOTAL_LAYERS)
    optimizer = torch.optim.Adam(router.parameters(), lr=LEARNING_RATE)

    # =========================================================================
    # Phase 1: Training Loop
    # =========================================================================
    print(divider())
    print(f"{Style.BOLD}{Style.WHITE}  PHASE 1{Style.RESET}{Style.DIM}  Training the SPA Router with Real Embeddings{Style.RESET}")
    print(divider())
    
    # We'll train on a long piece of code to get a variety of token embeddings.
    train_text = """
import torch
import torch.nn as nn
from transformers import AutoTokenizer

def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class CustomRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(896, 40)
"""
    train_embs, _ = get_token_embeddings(train_text)
    num_train_tokens = train_embs.size(0)
    print(f"  {Style.DIM}Training on {num_train_tokens} real tokens from sample code.{Style.RESET}")

    router.train()
    
    # We'll just loop over the available tokens for TRAIN_STEPS
    for step in range(TRAIN_STEPS):
        idx = step % num_train_tokens
        embedding = train_embs[idx]
        train_step(router, optimizer, embedding, CRITICAL_LAYERS, lam=LAMBDA)

    print(f"  {Style.GREEN}✓{Style.RESET} Training complete ({TRAIN_STEPS} steps).")
    print()

    # =========================================================================
    # Phase 2: C++ Inference Demo with Real Text
    # =========================================================================
    print(divider())
    print(f"{Style.BOLD}{Style.YELLOW}  PHASE 2{Style.RESET}{Style.DIM}  Language-Aware Inference via C++ Engine{Style.RESET}")
    print(divider())
    print()

    mgr = spa_engine.MemoryManager()
    mgr.set_verbose(False)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(script_dir, "..", WEIGHT_FILE)

    if not os.path.isfile(weight_path):
        print(f"{Style.RED}{Style.BOLD}  ERROR:{Style.RESET} Weight file not found: {weight_path}")
        sys.exit(1)

    # Test phrases ranging from simple syntax to complex logic
    test_phrases = [
        "import pandas as pd",             # Standard import
        "def quicksort(arr):",             # Algorithm def
        "    if len(arr) <= 1:",           # Simple condition
        "        return arr",              # Return
        "// A very standard comment line"  # Comment
    ]

    router.eval()
    
    for phrase_idx, text in enumerate(test_phrases):
        print(f"  {Style.BOLD}Phrase {phrase_idx}:{Style.RESET} '{Style.CYAN}{text}{Style.RESET}'")
        
        embs, input_ids = get_token_embeddings(text)
        
        # We process token by token, just like an autoregressive generation or processing step.
        for i in range(embs.size(0)):
            emb = embs[i]
            token_id = input_ids[i].item()
            token_str = tokenizer.decode([token_id])
            
            # 1. Complexity metric (L2 Norm of the embedding)
            complexity = torch.norm(emb, p=2).item()
            
            # 2. Predict Mask
            mask = router.predict(emb)
            num_compute = int(mask.sum().item())
            num_bypass = TOTAL_LAYERS - num_compute
            bypass_ratio = (num_bypass / TOTAL_LAYERS) * 100
            
            with torch.no_grad():
                soft_probs = router(emb)
            vis = mask_to_str(soft_probs, CRITICAL_LAYERS)

            # 3. Execute via C++ Engine
            t_start = time.perf_counter()
            for layer_idx in range(TOTAL_LAYERS):
                if mask[layer_idx].item() > 0.5:
                    ok = mgr.load_layer_mmap(weight_path)
                    if ok:
                        mgr.simulate_computation_pass()
                        mgr.unload_layer()
            t_end = time.perf_counter()
            elapsed_s = t_end - t_start
            
            # Print token stats
            # Clean up token string for display (replace newlines/spaces)
            display_str = token_str.replace('\n', '\\n').replace(' ', ' ')
            
            print(f"    {Style.DIM}Token:{Style.RESET} {display_str!r:<12} | "
                  f"{Style.DIM}Complexity (L2):{Style.RESET} {complexity:>6.2f} | "
                  f"{Style.DIM}Bypass:{Style.RESET} {bypass_ratio:>4.1f}% | "
                  f"{Style.DIM}Time:{Style.RESET} {elapsed_s:.4f}s")
            print(f"    {vis}")
        print()

    print(divider())
    print(f"{Style.DIM}  Engine shutting down...{Style.RESET}")
    print()

if __name__ == "__main__":
    main()
