import torch

# 假设有两个图像特征向量和两个文本特征向量
image_embeds = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
text_embeds = torch.tensor([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])

# 归一化处理
image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

print(image_embeds)
print(text_embeds)
from torch import nn
# ln = nn.LayerNorm()
# bn = nn.BatchNorm1d()

# NLP Example
batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)
layer_norm = nn.LayerNorm(embedding_dim) # : B L dim
# Activate module
print(layer_norm(embedding))

# Image Example
N, C, H, W = 20, 5, 10, 10
input = torch.randn(N, C, H, W)
# Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
# as shown in the image below
layer_norm = nn.LayerNorm([C, H, W])
output = layer_norm(input)
print(output)
