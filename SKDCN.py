import torch
import torch.nn as nn
import torch.nn.functional as F

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
        num_keywords=50,  # K: 关键词原型数量
        max_len=512,
        dropout=0.3,
        pretrained_embed=None,  # 可选：预训练嵌入矩阵 [vocab_size, embed_dim]
    ):
        super(SKDCN, self).__init__()

        self.embed_dim = embed_dim
        self.num_keywords = num_keywords

        # 1. 词嵌入层
        if pretrained_embed is not None:
            assert pretrained_embed.shape == (vocab_size, embed_dim)
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_embed), freeze=False
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 2. 位置编码（改进：使用随机初始化而非全零）
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.1)

        # 3. 语义编码通道：Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",  # 更平滑的激活函数
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 4. 可学习关键词库：K 个 d 维原型向量
        self.keyword_bank = nn.Parameter(torch.randn(num_keywords, embed_dim))
        nn.init.normal_(self.keyword_bank, mean=0.0, std=0.02)  # 小方差初始化

        # 5. 门控融合单元
        self.gate_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)  # 新增：融合后归一化（可选但推荐）

        # 6. 分类头
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        # ===== 1. Embedding + Positional Encoding =====
        x = self.embedding(input_ids)

        x = x + self.pos_encoding[:, :seq_len, :]

        src_key_padding_mask = (
            (~attention_mask.bool()) if attention_mask is not None else None
        )

        # ===== 2. Transformer Encoder =====
        h_sem = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # ===== 3. Semantic vector v_sem =====
        if attention_mask is not None:
            mask_float = attention_mask.unsqueeze(-1).float()
            v_sem = (h_sem * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(
                min=1.0
            )
        else:
            v_sem = h_sem.mean(dim=1)

        # ===== 4. Keyword channel =====
        # ===== 3. 关键词增强通道 =====
        x_norm = F.normalize(x, p=2, dim=-1)
        keyword_bank_norm = F.normalize(self.keyword_bank, p=2, dim=-1)
        sim = torch.matmul(x_norm, keyword_bank_norm.transpose(0, 1))  # [B, L, K]

        if attention_mask is not None:
            sim = sim.masked_fill(~attention_mask.unsqueeze(-1).bool(), float("-inf"))

        # 🔥 关键修复：防止全 -inf 导致 softmax(nan)
        is_inf = torch.isinf(sim)
        all_inf = is_inf.all(dim=-1, keepdim=True)  # [B, L, 1]
        sim = torch.where(all_inf, torch.zeros_like(sim), sim)

        A = F.softmax(sim, dim=-1)

        keyword_importance = A.max(dim=1).values
        v_key = torch.matmul(keyword_importance, self.keyword_bank)

        # ===== 5. Fusion & Output =====
        concat = torch.cat([v_sem, v_key], dim=-1)
        gate = torch.sigmoid(self.gate_proj(concat))
        v_fuse = gate * v_sem + (1 - gate) * v_key
        logits = self.classifier(v_fuse).squeeze(-1)

        return logits


# class SKDCN(nn.Module):
#     def __init__(
#         self,
#         vocab_size,
#         embed_dim=128,
#         num_heads=4,
#         num_layers=2,
#         hidden_dim=256,
#         num_keywords=50,  # K: 关键词原型数量
#         max_len=512,
#         dropout=0.3,
#         pretrained_embed=None,  # 可选：预训练嵌入矩阵 [vocab_size, embed_dim]
#     ):
#         super(SKDCN, self).__init__()

#         self.embed_dim = embed_dim
#         self.num_keywords = num_keywords

#         # 1. 词嵌入层
#         if pretrained_embed is not None:
#             assert pretrained_embed.shape == (vocab_size, embed_dim)
#             self.embedding = nn.Embedding.from_pretrained(
#                 torch.FloatTensor(pretrained_embed), freeze=False
#             )
#         else:
#             self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

#         # 位置编码（用于 Transformer）
#         self.pos_encoding = nn.Parameter(torch.zeros(1, max_len, embed_dim))

#         # 2. 语义编码通道：Transformer Encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim,
#             dropout=dropout,
#             batch_first=True,
#         )
#         self.transformer_encoder = nn.TransformerEncoder(
#             encoder_layer, num_layers=num_layers
#         )

#         # 3. 可学习关键词库：K 个 d 维原型向量
#         self.keyword_bank = nn.Parameter(torch.randn(num_keywords, embed_dim))
#         nn.init.xavier_uniform_(self.keyword_bank)

#         # 4. 门控融合单元
#         self.gate_proj = nn.Linear(embed_dim * 2, embed_dim)
#         self.dropout = nn.Dropout(dropout)

#         # 5. 分类头
#         self.classifier = nn.Linear(embed_dim, 1)

#     def forward(self, input_ids, attention_mask=None):
#         """
#         Args:
#             input_ids: [batch_size, seq_len] - token indices
#             attention_mask: [batch_size, seq_len] - 1 for real tokens, 0 for padding
#         Returns:
#             logits: [batch_size, 1] - raw output for BCEWithLogitsLoss
#         """
#         batch_size, seq_len = input_ids.shape

#         # ===== 1. Embedding + Positional Encoding =====
#         x = self.embedding(input_ids)  # [B, L, D]
#         x = x + self.pos_encoding[:, :seq_len, :]

#         # Apply attention mask for padding
#         if attention_mask is not None:
#             src_key_padding_mask = ~attention_mask.bool()  # True means ignore
#         else:
#             src_key_padding_mask = None

#         # ===== 2. 语义编码通道 =====
#         # Transformer expects [B, L, D]
#         h_sem = self.transformer_encoder(
#             x, src_key_padding_mask=src_key_padding_mask
#         )  # [B, L, D]

#         # Global average pooling over non-padded tokens
#         if attention_mask is not None:
#             # [B, L, 1] -> sum over L with mask
#             mask_float = attention_mask.unsqueeze(-1).float()
#             v_sem = (h_sem * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
#         else:
#             v_sem = h_sem.mean(dim=1)  # [B, D]

#         # ===== 3. 关键词增强通道 =====
#         # Compute similarity between each token and each keyword prototype
#         # x: [B, L, D], keyword_bank: [K, D] -> [B, L, K]
#         sim = torch.matmul(x, self.keyword_bank.transpose(0, 1)) / (
#             self.embed_dim**0.5
#         )  # scaled dot-product

#         # Apply attention mask to similarity
#         if attention_mask is not None:
#             sim = sim.masked_fill(~attention_mask.unsqueeze(-1).bool(), float("-inf"))

#         # Softmax over keywords (dim=-1): A_ij = softmax_j(sim_ij)
#         A = F.softmax(sim, dim=-1)  # [B, L, K]

#         # For each keyword j, take max_i A_ij across sequence
#         # This gives importance of keyword j in the whole document
#         keyword_importance = A.max(dim=1).values  # [B, K]

#         # Weighted sum of keyword prototypes
#         v_key = torch.matmul(keyword_importance, self.keyword_bank)  # [B, D]

#         # ===== 4. 动态门控融合 =====
#         concat = torch.cat([v_sem, v_key], dim=-1)  # [B, 2D]
#         gate = torch.sigmoid(self.gate_proj(concat))  # [B, D]
#         v_fuse = gate * v_sem + (1 - gate) * v_key  # [B, D]
#         v_fuse = self.dropout(v_fuse)

#         # ===== 5. 分类输出 =====
#         logits = self.classifier(v_fuse)  # [B, 1]
#         return logits
