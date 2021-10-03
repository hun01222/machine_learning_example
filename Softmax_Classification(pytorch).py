import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

x_data=[[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
y_data=[2,2,2,1,1,1,0,0]
z_data=[[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]
x_train=torch.FloatTensor(x_data)
y_train=torch.LongTensor(y_data)
z_train=torch.FloatTensor(z_data)

class softmaxClassifierModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear=nn.Linear(4, 3)
  def forward(self, x):
    return F.softmax(self.linear(x))

model=softmaxClassifierModel()
optimizer=optim.SGD(model.parameters(), lr=0.01)
nb_epochs=200001

for epoch in range(nb_epochs):
  prediction=model(x_train)
  loss=F.cross_entropy(prediction, y_train)
  pred=torch.argmax(prediction, 1)
  accuracy=(np.round(pred==y_train)).mean()
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if epoch%5000==0:
    print("Epoch {:4d} Accuracy: {:.3f}, Loss: {:.3f}".format(epoch, accuracy, loss))