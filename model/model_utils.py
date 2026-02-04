import torch
import torch.nn.functional as F
import torch.nn as nn
import math

# ------------ Adaptive Feature Fusion (AFF) ---------------
class AFF(nn.Module):
    def __init__(self, channels=128, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        # 1D convolution architecture for feature vector adaptation
        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )

        # Global attention mechanism
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        """
        Input dimension: (batch_size, feature_dim)
        Output dimension: (batch_size, feature_dim)
        """
        # Dimension consistency check
        assert x.size() == y.size(), f"Dimension mismatch: x{x.size()} vs y{y.size()}"

        # Convert to format required by 1D convolution (batch, channels, length)
        xy = ((x + y).unsqueeze(-1))  # [B, C] -> [B, C, 1]

        # Local attention branch
        xl = self.local_att(xy)  # [B, C, 1]
        # Global attention branch
        xg = self.global_att(xy)  # [B, C, 1]
        # Fusion weight calculation
        wei = self.sigmoid(xl + xg)  # [B, C, 1]
        wei = wei.squeeze(-1)  # [B, C]

        # Weighted fusion
        fused = x * wei + y * (1 - wei)  # [B, C]
        return fused


# ------------------------ Multi-head Bidirectional Cross-Attention (MBCA) -----------------------------
class MBCA(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attn_map = AttenMapNHeads(hidden_dim, num_heads)
        self.attention_fc_dp = nn.Linear(num_heads, hidden_dim)
        self.attention_fc_pd = nn.Linear(num_heads, hidden_dim)
        self.num_heads = num_heads  # Save number of heads for visualization

    def forward(self, drug, cell):
        attn_map = self.attn_map(drug, cell)

        att_dc = F.softmax(attn_map, dim=-1)  # [bs, nheads, d_len, p_len]
        att_cd = F.softmax(attn_map, dim=-2)  # [bs, nheads, d_len, p_len]
        attn_matrix = 0.5 * att_dc + 0.5 * att_cd  # [bs, nheads, d_len, p_len]

        drug_attn = self.attention_fc_dp(torch.mean(attn_matrix, -1).transpose(-1, -2))  # [bs, d_len, nheads]
        cell_attn = self.attention_fc_pd(torch.mean(attn_matrix, -2).transpose(-1, -2))  # [bs, p_len, nheads]

        drug_attn = F.sigmoid(drug_attn)
        cell_attn = F.sigmoid(cell_attn)

        drug = drug + drug * drug_attn
        cell = cell + cell * cell_attn

        drug, _ = torch.max(drug, 1)
        cell, _ = torch.max(cell, 1)

        pair = torch.cat([drug, cell], dim=1)

        return pair


class AttenMapNHeads(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()

        self.hid_dim = hidden_size
        self.n_heads = num_heads

        assert self.hid_dim % self.n_heads == 0

        self.f_q = nn.Linear(self.hid_dim, self.hid_dim)
        self.f_k = nn.Linear(self.hid_dim, self.hid_dim)
        self.d_k = self.hid_dim // self.n_heads

    def forward(self, d, p):
        batch_size = d.shape[0]

        Q = self.f_q(d)
        K = self.f_k(p)

        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        return attn_weights