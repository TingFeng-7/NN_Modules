import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.query = nn.Conv2d(in_channels=in_dim,
                                out_channels=in_dim,kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_dim,
                                out_channels=in_dim,kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_dim,
                                out_channels=in_dim,kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        bs, c , h, w = x.size()
        proj_q = self.query(x).reshape(
            bs, -1 , h*w).transpose(0, 2, 1)  # [bs, v_seq, c]
        proj_k = self.key(x).reshape(bs, -1, h*w) # [bs, c, v_seq]
        energy = torch.bmm(proj_q, proj_k) # 矩阵乘 [v_seq,c]*[c,v_seq]
        attention = self.softmax(energy)
        proj_value = self.value(x).reshape(bs, -1, w*h)
        
        out = torch.bmm(proj_value, attention.transpose(0, 2, 1))
        out = out.reshape(bs, c, h, w)

        return out


def main():
    attention_block = SelfAttention(64)
    input = torch.rand([4, 64, 32, 32])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
        
