import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(12)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

model = SimpleNet()
input_data = torch.randn(4, 3, 4, 4)# b c h w 

for name, param in model.bn.named_parameters(): # : 只能看到两个参数， ema的均值和方差是在推理阶段使用的
    print(f"{name}: {param}")
    
model_state_dict = model.state_dict()
for key in model_state_dict.keys():
    if "bn" in key:
        print(key)
        
output = model(input_data)
print(input_data[0][0])
print(output[0][0])
        