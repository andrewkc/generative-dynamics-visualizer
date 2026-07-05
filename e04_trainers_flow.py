from torch.optim import Adam

from models import FlowModel
from trainers import FlowTrainer
from datasets import get_dataset

dataset = get_dataset("two_moons")

model = FlowModel()

optimizer = Adam(
    model.parameters(),
    lr=1e-3,
)

trainer = FlowTrainer(
    model=model,
    optimizer=optimizer,
)

loss = trainer.train_epoch(
    dataset,
    steps_per_epoch=500,
    batch_size=256,
)

print(loss)