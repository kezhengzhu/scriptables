#!/usr/bin/env python
import torch
import numpy as np 
import pandas as pd 
from torch import nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt 

from sklearn import preprocessing

# This is a python source code used to document all kinds of machine learning techniques
# The functions here should be used as a precursor for future packaging of ML frameworks
# for thermo applications

class SimpleNet(nn.Module):
    def __init__(self, input_output, *args):
        super().__init__()
        input_dim = input_output[0]
        output_dim = input_output[1]
        self.beta = nn.ModuleList()
        self.sig = nn.Softplus()
        for i in range(len(args)):
            if i == 0:
                self.beta.append(nn.Linear(input_dim, args[i]))

            if i < len(args)-1:
                self.beta.append(nn.Linear(args[i], args[i+1]))
            else:
                self.beta.append(nn.Linear(args[i],output_dim))
        self.layers = len(self.beta)


    def forward(self, x):
        for i in range(self.layers):
            f = self.beta[i]
            if i == 0:
                x = f(x)
            else:
                x = self.sig(f(x))
        return x

df = pd.read_csv('PvTdata.csv', header=None)
df = df[df.iloc[:,4] <= 30]
dataset = df.values
v = dataset[:,0]
rho = dataset[:,1]
T = dataset[:,2]
p_eos = dataset[:,3]
p_real = dataset[:,4]

k = 1.38064852e-23
Na = 6.02214086e23
z = p_real * 1e5 / (Na * k * rho * T)
data = np.zeros((dataset.shape[0],dataset.shape[1]+1))
dataset = np.column_stack((v,T, p_eos, p_real, z))

validation = np.column_stack([np.copy(dataset[0:1000,:]),np.copy(dataset[80000:81000,:]),np.copy(dataset[140000:141000,:])])

np.random.shuffle(dataset)

v = dataset[:,0]
T = dataset[:,1]
p_eos = dataset[:,2]
p_real = dataset[:,3]
z = dataset[:,4]


X = np.column_stack([np.log(v), T])
# Y = p_eos.reshape((np.size(p_eos),1))
Y = p_real.reshape((np.size(p_real),1))

XV = np.column_stack([np.log(validation[:,0]), validation[:,1]])
YV = validation[:,3:4]

print(X.shape, Y.shape, XV.shape, YV.shape)

scalerx = preprocessing.StandardScaler()
scaledX = scalerx.fit_transform(X)
scaledXV = scalerx.transform(XV)
print(scalerx, scalerx.mean_, scalerx.scale_)

scalery = preprocessing.StandardScaler()
scaledY = scalery.fit_transform(Y)
scaledYV = scalery.transform(YV)
print(scalery, scalery.mean_, scalery.scale_)


X_t = torch.from_numpy(scaledX).float()
Y_t = torch.from_numpy(Y).float()

XV_t = torch.from_numpy(scaledXV).float()
YV_t = torch.from_numpy(YV).float()

model = SimpleNet((2,1), *[200],20)
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

n_epochs = 1000

for epoch in range(n_epochs):
    ### Training step
    model.train()
    optimizer.zero_grad()

    Y_ = model(X_t)
    loss = criterion(Y_, Y_t)

    print('epoch: ', epoch,' loss: ', loss.item())
    loss.backward()
    optimizer.step()

### Eval
model.eval()
with torch.no_grad():
    Y_ = model(XV_t)
    loss = criterion(Y_, YV_t)


fig, ax = plt.subplots()
ax.plot(scalerx.inverse_transform(XV_t.numpy()[:,0]), scalery.inverse_transform(Y_.numpy()), '.', label='pred')
ax.plot(scalerx.inverse_transform(XV_t.numpy()[:,0]), scalery.inverse_transform(YV_t.numpy()), '.', label='data')

ax.set_title(f"MSE: {loss.item():0.5f}")
ax.legend()
plt.show()

torch.save(model, "p_from_vt.pth")
