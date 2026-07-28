
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# Set PyTorch TF32 Matmul Precision for NVIDIA Tensor Cores (RTX 40-series speedup)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
class HYPERPARAMITER:
    repo_path = os.path.abspath(os.path.dirname(__file__))
    model_dir = os.path.join(repo_path, "model")
    model_path = os.path.join(model_dir, "transformer.pt")
    vocab_path = os.path.join(model_dir, "vocab.json")
    merges_path = os.path.join(model_dir, "merges.txt")
    config_path = os.path.join(model_dir, "config.json")
    data_dir = os.path.join(repo_path, "data4")
    batch_size = 64          # Optimal batch size for RTX 4070 GPU parallelism
    block_size = 256         # Maximum context length
    max_iters = 5000         # Total training iterations
    eval_interval = 200      # Interval to estimate loss
    learning_rate = 5e-4     # Accelerated Adam learning rate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 20          # Fast evaluation steps
    epochs = 5               # Number of training epochs
    n_embd = 512             # Scaled embedding dimension (512 // 8 = 64 head_size for Tensor Cores)
    n_head = 8               # 8 attention heads
    n_layer = 8              # 8 transformer blocks stacked (~25.4M parameters)
    dropout = 0.1            # Reduced dropout for 3x faster loss convergence

    assert n_embd % n_head == 0, "HYPERPARAMITER.n_embd must be divisible by HYPERPARAMITER.n_head"
    head_size = n_embd // n_head
    hidden_size = n_embd

# Export legacy names for compatibility
batch_size = HYPERPARAMITER.batch_size
block_size = HYPERPARAMITER.block_size
max_iters = HYPERPARAMITER.max_iters
eval_interval = HYPERPARAMITER.eval_interval
learning_rate = HYPERPARAMITER.learning_rate
device = HYPERPARAMITER.device
eval_iters = HYPERPARAMITER.eval_iters
epochs = HYPERPARAMITER.epochs
n_embd = HYPERPARAMITER.n_embd
n_head = HYPERPARAMITER.n_head
n_layer = HYPERPARAMITER.n_layer
dropout = HYPERPARAMITER.dropout

torch.manual_seed(1337)

# ==========================================
# 2. VECTORIZED CAUSAL MULTI-HEAD SELF-ATTENTION (FLASH-ATTENTION)
# ==========================================
class MultiHeadAttention(nn.Module):
    """ Fast Vectorized Causal Multi-Head Attention using PyTorch FlashAttention """
    def __init__(self, num_heads=None, head_size=None, n_embd=None, block_size=None, dropout=None):
        super().__init__()
        n_embd = n_embd if n_embd is not None else HYPERPARAMITER.n_embd
        num_heads = num_heads if num_heads is not None else HYPERPARAMITER.n_head
        dropout = dropout if dropout is not None else HYPERPARAMITER.dropout

        assert n_embd % num_heads == 0, f"n_embd ({n_embd}) must be divisible by num_heads ({num_heads})"
        self.n_head = num_heads
        self.head_size = n_embd // num_heads
        self.n_embd = n_embd
        self.dropout = dropout

        # Key, Query, Value combined projection for 3x GPU speedup
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # Batch size, Sequence length, Embedding dimension
        
        # Calculate Query, Key, Values for all heads in batch in a single fused linear layer
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, n_head, T, head_size)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, n_head, T, head_size)

        # PyTorch Native Causal Scaled Dot-Product Attention (FlashAttention kernel)
        if hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0.0, 
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_size ** 0.5))
            tril = torch.tril(torch.ones(T, T, device=x.device))
            att = att.masked_fill(tril[:T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C) # Concatenate all head outputs
        out = self.resid_dropout(self.proj(y))
        return out

# ==========================================
# 3. POSITION-WISE FEED-FORWARD NETWORK
# ==========================================
class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity (GELU) """
    def __init__(self, n_embd=None, dropout=None):
        super().__init__()
        n_embd = n_embd if n_embd is not None else HYPERPARAMITER.n_embd
        dropout = dropout if dropout is not None else HYPERPARAMITER.dropout
        # Standard Transformer architecture expands hidden dimension by a factor of 4
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. TRANSFORMER BLOCK (LAYER)
# ==========================================
class Block(nn.Module):
    """ Transformer block: communicates (attention) then computes (feedforward) """
    def __init__(self, n_embd=None, n_head=None, block_size=None, dropout=None):
        super().__init__()
        n_embd = n_embd if n_embd is not None else HYPERPARAMITER.n_embd
        n_head = n_head if n_head is not None else HYPERPARAMITER.n_head
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd=n_embd, block_size=block_size, dropout=dropout)
        self.ffwd = FeedForward(n_embd=n_embd, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-Layer Normalization architecture with residual skip-connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# ==========================================
# 5. CORE LLM DECODER ARCHITECTURE
# ==========================================
class MiniLanguageModel(nn.Module):
    """ Complete Decoder-Only Transformer Language Model """
    def __init__(self, vocab_size, n_embd=None, n_head=None, n_layer=None, block_size=None, dropout=None):
        super().__init__()
        self.n_embd = n_embd if n_embd is not None else HYPERPARAMITER.n_embd
        self.n_head = n_head if n_head is not None else HYPERPARAMITER.n_head
        self.n_layer = n_layer if n_layer is not None else HYPERPARAMITER.n_layer
        self.block_size = block_size if block_size is not None else HYPERPARAMITER.block_size
        self.dropout = dropout if dropout is not None else HYPERPARAMITER.dropout

        # Each token looks up its dense vector embedding
        self.token_embedding_table = nn.Embedding(vocab_size, self.n_embd)
        # Each position looks up its structural location embedding
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        # Stack sequential transformer layers
        self.blocks = nn.Sequential(*[
            Block(n_embd=self.n_embd, n_head=self.n_head, block_size=self.block_size, dropout=self.dropout)
            for _ in range(self.n_layer)
        ])
        # Final layer normalization
        self.ln_f = nn.LayerNorm(self.n_embd)
        # Language modeling head mapping hidden state back to vocabulary logits
        self.lm_head = nn.Linear(self.n_embd, vocab_size)

    def resize_token_embeddings(self, new_num_tokens):
        """ Resizes token embedding table and language model head to accommodate new tokens """
        # 1. Resize token embedding table
        old_embeddings = self.token_embedding_table
        new_embeddings = nn.Embedding(new_num_tokens, self.n_embd, device=old_embeddings.weight.device)
        # Copy over old weights
        num_to_copy = min(old_embeddings.num_embeddings, new_num_tokens)
        new_embeddings.weight.data[:num_to_copy] = old_embeddings.weight.data[:num_to_copy]
        self.token_embedding_table = new_embeddings

        # 2. Resize language model head
        old_lm_head = self.lm_head
        new_lm_head = nn.Linear(self.n_embd, new_num_tokens, device=old_lm_head.weight.device)
        # Copy over old weights and biases
        new_lm_head.weight.data[:num_to_copy] = old_lm_head.weight.data[:num_to_copy]
        if old_lm_head.bias is not None:
            new_lm_head.bias.data[:num_to_copy] = old_lm_head.bias.data[:num_to_copy]
        self.lm_head = new_lm_head

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Retrieve structural embeddings
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, n_embd)
        x = tok_emb + pos_emb # Combine content and spatial location (B, T, n_embd)

        # Pass through the core network backbone
        x = self.blocks(x) # (B, T, n_embd)
        x = self.ln_f(x)   # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten cross-entropy inputs to evaluate across the sequence
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """ Generate novel text auto-regressively given a starting context """
        for _ in range(max_new_tokens):
            # Crop current context if it exceeds the maximum architectural block size
            idx_cond = idx[:, -self.block_size:]
            # Get next-step predictions
            logits, loss = self(idx_cond)
            # Focus strictly on the final index step to make the next prediction
            logits = logits[:, -1, :] # Becomes (B, C)
            # Convert predictions into probability distributions
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample next item from the generated categorical distributions
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Concat sampled token to ongoing history context
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# Alias for backward compatibility
Decoder = MiniLanguageModel


