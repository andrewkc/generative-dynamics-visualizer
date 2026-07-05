import torch

from models import DiffusionModel

model = DiffusionModel()

x = torch.randn(8, 2)

t = torch.rand(8)

y = model(x, t)

print(y.shape)