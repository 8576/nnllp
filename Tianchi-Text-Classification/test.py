import torch.nn.functional as F
import torch

input = torch.randn(3, 5, requires_grad=True)
target = torch.randint(5, (3,), dtype=torch.int64)
print(input, target, sep="\n")

loss = F.cross_entropy(input, target)
loss.backward()

# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
print(input, target, sep="\n")
loss = F.cross_entropy(input, target)
loss.backward()