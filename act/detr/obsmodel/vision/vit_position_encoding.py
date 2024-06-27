import torch
import torch.nn as nn
import math

class PositionEmbeddingSine1D(nn.Module):
    """
    Position Embedding Sine for 1D sequences (e.g., output of ViT)
    """
    def __init__(self, num_pos_feats, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # 通常设为特征维数的一半
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):  # (8.197,768)
        n = x.shape[1]  # 序列长度
        device = x.device
        position = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = self.temperature ** (2 * torch.arange(self.num_pos_feats, dtype=torch.float32, device=device) / self.num_pos_feats)  # (384)
        pos = position / div_term  # (197,384)

        pos = torch.cat((pos.sin(), pos.cos()), dim=1)  # (197,768)
        # print('vit',self.normalize)
        # if self.normalize:
        #     eps = 1e-6
        #     pos = pos / (pos.max() + eps) * self.scale

        pos = pos.unsqueeze(0)  # 保持 batch size 为 1
        return pos  # 输出形状 (1, N, D) (1,197,384)


class PositionEmbeddingLearned(nn.Module):
    """
    Learned position embedding for 1D sequences.
    """
    def __init__(self, num_pos_feats, max_len=500):
        super().__init__()
        self.row_embed = nn.Embedding(max_len, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight, -0.05, 0.05)

    def forward(self, x):
        n = x.shape[1]  # 序列长度
        device = x.device
        positions = torch.arange(n, device=device)
        pos_embed = self.row_embed(positions)
        pos_embed = pos_embed.unsqueeze(0)  # 保持 batch size 为 1
        return pos_embed  # 输出形状 (1, N, D)



def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine1D(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding





# 测试代码
batch_size = 5
sequence_length = 197
feature_dim = 768
tensor = torch.randn(batch_size, sequence_length, feature_dim)
pos_encoder = PositionEmbeddingSine1D(num_pos_feats=feature_dim)
pos_encoding = pos_encoder(tensor)
print(pos_encoding.shape)  # 输出 (1, 197, 768)，可自动广播到与输入 tensor 相同的 batch size