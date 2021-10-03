import torch
import torch.optim as optim
import numpy as np

x_data=[[1,2],[2,3],[3,4],[4,3],[5,3],[6,2]]
y_data=[[0],[0],[0],[1],[1],[1]]
x_train=torch.FloatTensor(x_data).float()
y_train=torch.FloatTensor(y_data).float()
W=torch.zeros((2,1), requires_grad=True)
b=torch.zeros(1, requires_grad=True)
optimizer=optim.SGD([W,b], lr=0.01)
nb_epochs=2001

for epoch in range(nb_epochs):
  binary_classification=torch.sigmoid(x_train.matmul(W)+b)
  cost=-(y_train*torch.log(binary_classification)+(1-y_train)*torch.log(1-binary_classification)).mean()
  prediction=np.round(binary_classification>0.5).type(torch.float32)
  accuracy=(np.round(prediction==y_train)).mean()
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()
  if epoch%100==0:
    print("Epoch: ", epoch, "/", nb_epochs, "Model: ", binary_classification.detach().numpy(), "correct: ", prediction.detach().numpy(), accuracy.item(), "Cost: \n", cost.item())