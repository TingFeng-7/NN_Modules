import torch

# 假设有两个图像特征向量和两个文本特征向量
image_embeds = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
text_embeds = torch.tensor([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])

# 归一化处理
image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

print(image_embeds)
print(text_embeds)