import torch
import torch.nn as nn
import torch.nn.functional as F


class SKDCN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        hidden_dim=256,
        num_keywords=50,
        max_len=512,
        dropout=0.3,
        pretrained_embed=None,
    ):
        super(SKDCN, self).__init__()

        self.embed_dim = embed_dim
        self.num_keywords = num_keywords

        if pretrained_embed is not None:
            assert pretrained_embed.shape == (vocab_size, embed_dim)
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_embed), freeze=False
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.keyword_bank = nn.Parameter(torch.randn(num_keywords, embed_dim))
        nn.init.normal_(self.keyword_bank, mean=0.0, std=0.02)

        self.gate_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        x = self.embedding(input_ids)

        x = x + self.pos_encoding[:, :seq_len, :]

        src_key_padding_mask = (
            (~attention_mask.bool()) if attention_mask is not None else None
        )

        h_sem = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        if attention_mask is not None:
            mask_float = attention_mask.unsqueeze(-1).float()
            v_sem = (h_sem * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(
                min=1.0
            )
        else:
            v_sem = h_sem.mean(dim=1)

        x_norm = F.normalize(x, p=2, dim=-1)
        keyword_bank_norm = F.normalize(self.keyword_bank, p=2, dim=-1)
        sim = torch.matmul(x_norm, keyword_bank_norm.transpose(0, 1))

        if attention_mask is not None:
            sim = sim.masked_fill(~attention_mask.unsqueeze(-1).bool(), float("-inf"))
        is_inf = torch.isinf(sim)
        all_inf = is_inf.all(dim=-1, keepdim=True)  # [B, L, 1]
        sim = torch.where(all_inf, torch.zeros_like(sim), sim)

        A = F.softmax(sim, dim=-1)

        keyword_importance = A.max(dim=1).values
        v_key = torch.matmul(keyword_importance, self.keyword_bank)

        concat = torch.cat([v_sem, v_key], dim=-1)
        gate = torch.sigmoid(self.gate_proj(concat))
        v_fuse = gate * v_sem + (1 - gate) * v_key
        logits = self.classifier(v_fuse).squeeze(-1)

        return logits
