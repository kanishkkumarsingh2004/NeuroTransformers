
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

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
    data_dir = os.path.join(repo_path, "data1")
    # data_path = os.path.join(data_dir, "input.txt")
    batch_size = 16          # Number of independent sequences processed in parallel
    block_size = 256         # Maximum context length (window size)
    max_iters = 5000          # Total training iterations
    eval_interval = 500       # Intentional interval to estimate loss
    learning_rate = 3e-4     # Adam learning rate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    epochs = 1              # Number of training epochs
    n_embd = 1000            # Embedding dimension size
    n_head = 10               # Number of attention heads (must divide n_embd evenly)
    n_layer =8             # Number of transformer blocks stacked
    dropout = 0.2            # Dropout probability

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
# 2. CAUSAL MULTI-HEAD SELF-ATTENTION
# ==========================================
class Head(nn.Module):
    """ Single head of causal self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Register a lower-triangular causal mask buffer (not a trainable parameter)
        self.register_buffer('tril', torch.tril(torch.ones(HYPERPARAMITER.block_size, HYPERPARAMITER.block_size)))
        self.dropout = nn.Dropout(HYPERPARAMITER.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # Compute attention scores ("affinities") scaled by the square root of head size
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # Mask future tokens to prevent the model from looking ahead
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Perform the weighted aggregation of values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention running in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(HYPERPARAMITER.n_embd, HYPERPARAMITER.n_embd) # Projection layer back into residual pathway
        self.dropout = nn.Dropout(HYPERPARAMITER.dropout)

    def forward(self, x):
        # Concatenate outputs from all heads along the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# ==========================================
# 3. POSITION-WISE FEED-FORWARD NETWORK
# ==========================================
class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity (GELU) """
    def __init__(self, n_embd):
        super().__init__()
        # Standard Transformer architecture expands hidden dimension by a factor of 4
        self.net = nn.Sequential(
            nn.Linear(HYPERPARAMITER.n_embd, 4 * HYPERPARAMITER.n_embd),
            nn.GELU(),
            nn.Linear(4 * HYPERPARAMITER.n_embd, HYPERPARAMITER.n_embd),
            nn.Dropout(HYPERPARAMITER.dropout),
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. TRANSFORMER BLOCK (LAYER)
# ==========================================
class Block(nn.Module):
    """ Transformer block: communicates (attention) then computes (feedforward) """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
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
    def __init__(self, vocab_size):
        super().__init__()
        # Each token looks up its dense vector embedding
        self.token_embedding_table = nn.Embedding(vocab_size, HYPERPARAMITER.n_embd)
        # Each position looks up its structural location embedding
        self.position_embedding_table = nn.Embedding(HYPERPARAMITER.block_size, HYPERPARAMITER.n_embd)
        # Stack sequential transformer layers
        self.blocks = nn.Sequential(*[Block(HYPERPARAMITER.n_embd, n_head=HYPERPARAMITER.n_head) for _ in range(HYPERPARAMITER.n_layer)])
        # Final layer normalization
        self.ln_f = nn.LayerNorm(HYPERPARAMITER.n_embd)
        # Language modeling head mapping hidden state back to vocabulary logits
        self.lm_head = nn.Linear(HYPERPARAMITER.n_embd, vocab_size)

    def resize_token_embeddings(self, new_num_tokens):
        """ Resizes token embedding table and language model head to accommodate new tokens """
        # 1. Resize token embedding table
        old_embeddings = self.token_embedding_table
        new_embeddings = nn.Embedding(new_num_tokens, HYPERPARAMITER.n_embd, device=old_embeddings.weight.device)
        # Copy over old weights
        num_to_copy = min(old_embeddings.num_embeddings, new_num_tokens)
        new_embeddings.weight.data[:num_to_copy] = old_embeddings.weight.data[:num_to_copy]
        self.token_embedding_table = new_embeddings

        # 2. Resize language model head
        old_lm_head = self.lm_head
        new_lm_head = nn.Linear(HYPERPARAMITER.n_embd, new_num_tokens, device=old_lm_head.weight.device)
        # Copy over old weights and biases
        new_lm_head.weight.data[:num_to_copy] = old_lm_head.weight.data[:num_to_copy]
        if old_lm_head.bias is not None:
            new_lm_head.bias.data[:num_to_copy] = old_lm_head.bias.data[:num_to_copy]
        self.lm_head = new_lm_head

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Retrieve structural embeddings
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=HYPERPARAMITER.device)) # (T, n_embd)
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
            idx_cond = idx[:, -HYPERPARAMITER.block_size:]
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


