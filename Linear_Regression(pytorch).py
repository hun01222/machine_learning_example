import torch
import torch.optim as optim
import numpy as np

x_data=torch.FloatTensor([[1,1], [2,2], [3,3]])
y_data=torch.FloatTensor([[10], [20], [30]])
W=torch.zeros([2,1], requires_grad=True)
b=torch.zeros([1], requires_grad=True)
optimizer=optim.SGD([W, b], lr=0.01)

for epoch in range(2001):
  model=torch.matmul(x_data, W)+b
  cost=torch.mean((model-y_data)**2)
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

print(W)
print(b)
print(W.detach().numpy())
print(b.detach().numpy())