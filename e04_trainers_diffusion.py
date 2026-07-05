from torch.optim import Adam

from models import DiffusionModel
from diffusion import get_sde
from trainers import DiffusionTrainer
from datasets import get_dataset

dataset = get_dataset("two_moons")

model = DiffusionModel()

optimizer = Adam(
    model.parameters(),
    lr=1e-3,
)

sde = get_sde("vp")

trainer = DiffusionTrainer(
    model=model,
    sde=sde,
    optimizer=optimizer,
    prediction_type="epsilon",
)

loss = trainer.train_epoch(
    dataset,
    steps_per_epoch=500,
    batch_size=256,
)

print(loss)