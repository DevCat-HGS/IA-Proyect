import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.query(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(out)

class MainModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1000, d_model))
        
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.final = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x)
        x = x + self.pos_encoding[:seq_len]
        x = self.dropout(x)
        
        for attention in self.attention_layers:
            residual = x
            x = self.layer_norm(attention(x, x, x) + residual)
            
        return self.final(x)

    def generate(self, start_tokens, max_len=100, temperature=1.0):
        self.eval()
        current_seq = start_tokens
        
        for _ in range(max_len):
            with torch.no_grad():
                logits = self(current_seq)
                next_token_logits = logits[:, -1, :] / temperature
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
                current_seq = torch.cat([current_seq, next_token], dim=1)
                
        return current_seq
