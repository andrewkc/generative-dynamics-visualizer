import torch

from flow_matching import (
    LinearInterpolation,
    ConstantVectorField,
    flow_matching_loss,
)

torch.manual_seed(42)

batch_size = 8

#
# Datos de prueba
#

x0 = torch.randn(batch_size, 2)

x1 = torch.randn(batch_size, 2)

t = torch.rand(batch_size)

#
# Interpolación
#

interpolation = LinearInterpolation()

xt = interpolation.sample(
    x0,
    x1,
    t,
)

print("Interpolated shape :", xt.shape)

#
# Campo de velocidades
#

field = ConstantVectorField()

velocity = field.target(
    x0,
    x1,
)

print("Velocity shape     :", velocity.shape)

#
# Loss
#

prediction = velocity + 0.1 * torch.randn_like(velocity)

loss = flow_matching_loss(
    prediction,
    velocity,
)

print("Loss               :", loss.item())