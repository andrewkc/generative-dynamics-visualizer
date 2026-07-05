import torch

from torch.optim import Adam

from datasets import get_dataset

from models import FlowModel

from trainers import FlowTrainer

dataset = get_dataset("two_moons")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = FlowModel().to(device)

optimizer = Adam(
    model.parameters(),
    lr=1e-3,
)

trainer = FlowTrainer(
    model=model,
    optimizer=optimizer,
    device=device,
)

loss = trainer.train_epoch(
    dataset=dataset,
    steps_per_epoch=500,
    batch_size=256,
)

print()

print("Average loss:", loss)