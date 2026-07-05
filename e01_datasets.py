from datasets import get_dataset

dataset = get_dataset("eight_gaussians")

x = dataset.sample(5000)

print(x.shape)

# torch.Size([5000,2])