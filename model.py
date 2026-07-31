import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

# Enable TensorCore acceleration for FP32/TF32 operations
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

# ==========================================
# 1. MODEL CONFIGURATION
# ==========================================
class ModelConfig:
    # Project Paths
    repo_path: str = os.path.abspath(os.path.dirname(__file__))
    model_dir: str = os.path.join(repo_path, "model")
    model_path: str = os.path.join(model_dir, "transformer.pt")
    vocab_path: str = os.path.join(model_dir, "vocab.json")
    merges_path: str = os.path.join(model_dir, "merges.txt")
    config_path: str = os.path.join(model_dir, "config.json")
    data_dir: str = os.path.join(repo_path, "data4")

    # Training Runtime Settings
    batch_size: int = 32
    block_size: int = 256  # Kept as alias for max_seq_len
    eval_iters: int = 20
    eval_interval: int = 200
    learning_rate: float = 5e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(
        self,
        vocab_size: int = 2048,
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        tie_word_embeddings: bool = True,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = dim // n_heads
        self.max_seq_len = max_seq_len
        self.block_size = max_seq_len  # Alias so data.py works seamlessly
        self.dropout = dropout
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings

        if hidden_dim is None:
            hidden_dim = int(2 * (4 * dim) / 3)
            self.hidden_dim = ((hidden_dim + 255) // 256) * 256
        else:
            self.hidden_dim = hidden_dim


# ==========================================
# 2. MODERN LLM BUILDING BLOCKS
# ==========================================
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA style)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequency tensor
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin caches for max sequence length
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_heads, seq_len, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, head_dim)
        return (x * cos) + (self._rotate_half(x) * sin)


class SwiGLUFeedForward(nn.Module):
    """SwiGLU Feed-Forward Network (Gated Linear Units)"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)  # Down projection
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)  # Up projection
        self.dropout = nn.Dropout(config.dropout)

        # BUG FIX: _init_weights() below checks for this flag to apply
        # GPT-2/nanoGPT-style depth scaling on residual-stream projections,
        # but nothing ever set it -- the scaling never fired on any layer.
        # w2 writes directly back into the residual stream, so it's the one
        # that needs scaling down as depth grows (unscaled residual writes
        # compound across layers and can destabilize deep stacks).
        self.w2.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (Swish(w1(x)) * w3(x)) -> w2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class GroupedQueryAttention(nn.Module):
    """Multi-Head / Grouped-Query Attention with FlashAttention and KV-Cache support"""
    def __init__(self, config: ModelConfig, rope: RotaryEmbedding):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.num_key_value_groups = self.n_heads // self.n_kv_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.rope = rope

        self.q_proj = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Same fix as SwiGLUFeedForward.w2 -- out_proj also writes directly
        # into the residual stream and needs the depth-scaled init to fire.
        self.out_proj.NANOGPT_SCALE_INIT = 1

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(q, T)
        q = self.rope.apply_rope(q, cos, sin)
        k = self.rope.apply_rope(k, cos, sin)

        # Handle KV-Caching during inference
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=-2)
            v = torch.cat([v_prev, v], dim=-2)
        new_kv_cache = (k, v) if not self.training else None

        # Repeat KV heads for Grouped Query Attention if n_kv_heads < n_heads
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # High-Performance Scaled Dot-Product Attention (FlashAttention-2 Kernel)
        is_causal = True if kv_cache is None else False
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.resid_dropout(self.out_proj(y))
        return out, new_kv_cache


class TransformerBlock(nn.Module):
    """Decoder Transformer Block with Pre-RMSNorm"""
    def __init__(self, config: ModelConfig, rope: RotaryEmbedding):
        super().__init__()
        self.attn = GroupedQueryAttention(config, rope)
        self.ffn = SwiGLUFeedForward(config)
        self.input_layernorm = RMSNorm(config.dim, eps=config.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # 1. Attention Block with Pre-RMSNorm and Residual Connection
        normed_attn_in = self.input_layernorm(x)
        attn_out, new_kv_cache = self.attn(normed_attn_in, kv_cache=kv_cache)
        x = x + attn_out

        # 2. Feed-Forward Block with Pre-RMSNorm and Residual Connection
        normed_ffn_in = self.post_attention_layernorm(x)
        ffn_out = self.ffn(normed_ffn_in)
        x = x + ffn_out

        return x, new_kv_cache


# ==========================================
# 3. PRODUCTION TRANSFORMER LLM DECODER
# ==========================================
class ModernLLM(nn.Module):
    """Production Grade Decoder-Only Language Model Architecture"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.rope = RotaryEmbedding(config.head_dim, max_seq_len=config.max_seq_len, theta=config.rope_theta)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config, self.rope) for _ in range(config.n_layers)
        ])
        
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight Tying (Shares input embeddings with output lm_head)
        if config.tie_word_embeddings:
            self.output.weight = self.tok_embeddings.weight

        # Parameter Initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Scaled Xavier/Normal Weight Initialization"""
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        
        B, T = idx.shape
        x = self.tok_embeddings(idx)

        new_kv_caches = [] if kv_caches is not None else None

        for i, layer in enumerate(self.layers):
            layer_kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = layer(x, kv_cache=layer_kv)
            if new_kv_caches is not None:
                new_kv_caches.append(new_kv)

        x = self.norm(x)
        logits = self.output(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Efficient auto-regressive generation using KV-Caching.

        top_p: nucleus sampling -- keep the smallest set of tokens whose
        cumulative probability exceeds top_p, zero out the rest. Standard
        in modern LLM inference (used alongside or instead of top_k)
        because it adapts to how peaked/flat the distribution is at each
        step, unlike a fixed top_k count.

        repetition_penalty: >1.0 discourages repeating tokens already in
        the sequence (divides their logits by the penalty before
        sampling). Small models are especially prone to repetition loops,
        so this matters more here than it would on a much larger model.
        """
        self.eval()
        kv_caches = [None] * len(self.layers)

        # Initial prefill pass
        logits, _, kv_caches = self(idx, kv_caches=kv_caches)

        for _ in range(max_new_tokens):
            # Focus on the last token logits
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    seen = torch.unique(idx[b])
                    seen_logits = logits[b, seen]
                    logits[b, seen] = torch.where(
                        seen_logits > 0, seen_logits / repetition_penalty, seen_logits * repetition_penalty
                    )

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cum_probs > top_p
                # Always keep at least the single most probable token, so we
                # never zero out every candidate for a very peaked distribution.
                sorted_mask[..., 0] = False
                mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_idx, sorted_mask)
                logits = logits.masked_fill(mask, -float("Inf"))

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            # Subsequent decode passes only process the newest token using KV Cache
            logits, _, kv_caches = self(idx_next, kv_caches=kv_caches)

        return idx


device = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================
# 4. EXPORTS & BACKWARD COMPATIBILITY
# ==========================================
# Aliases for legacy imports across training and evaluation scripts
MiniLanguageModel = ModernLLM
Decoder = ModernLLM

# BUG FIX: this used to be `HYPERPARAMITER = ModelConfig` (the class itself,
# never instantiated). Every `getattr(HYPERPARAMITER, "vocab_size", ...)` /
# `"dim"` / `"n_layers"` / `"n_heads"` / `"dropout"` / `"max_seq_len"` lookup
# across the codebase was silently missing (those only exist on *instances*,
# set in __init__) and quietly falling back to each call site's hardcoded
# default instead of whatever you configured. Instantiating it here means
# constructor defaults (and the env overrides below) actually take effect.
HYPERPARAMITER = ModelConfig()

# Allow the active dataset directory / checkpoint path to be chosen per
# training run (e.g. `NEUROTRANSFORMERS_DATA_DIR=data3 python continuous_training.py`)
# instead of hardcoding a single folder you have to hand-edit every session.
HYPERPARAMITER.data_dir = os.environ.get("NEUROTRANSFORMERS_DATA_DIR", HYPERPARAMITER.data_dir)
HYPERPARAMITER.model_path = os.environ.get("NEUROTRANSFORMERS_MODEL_PATH", HYPERPARAMITER.model_path)  