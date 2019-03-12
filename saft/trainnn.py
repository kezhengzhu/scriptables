import torch
import numpy as np 
from torch import nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt 

from sklearn.datasets import make_regression

class TestNet(nn.Module):
    def __init__(self, input_dim, H1, H2, H3, H4, H5, H6):
        super().__init__()
        self.beta = nn.Linear(input_dim, H1)
        self.beta2 = nn.Linear(H1, H2)
        self.beta3 = nn.Linear(H2, H3)
        self.beta4 = nn.Linear(H3, H4)
        self.beta5 = nn.Linear(H4, H5)
        self.beta6 = nn.Linear(H5, H6)
        self.beta7 = nn.Linear(H6, 1)

    def forward(self, x):
        x = self.beta(x)
        x = F.relu(self.beta2(x))
        x = F.relu(self.beta3(x))
        x = F.relu(self.beta4(x))
        x = F.relu(self.beta5(x))
        x = F.relu(self.beta6(x))
        x = F.relu(self.beta7(x))

        return x

def test_func(x):
    return (1-np.exp(-0.0213*np.sqrt(x))) * np.sin(0.183*x**1.4/np.pi) * x**0.325 + np.cos(0.573*x**0.32/np.pi) * np.exp(-0.122*np.sqrt(x+x**2)) * x ** 0.539

def main():
    # n_features = 1
    # n_samples = 5000

    # x,y = make_regression(n_samples=n_samples, n_features=n_features, noise=10)
    # print(x.shape, y.shape)
    # fix, ax = plt.subplots()
    # ax.plot(x,y,'.')
    # plt.show()
    
    nsamples = 200000
    x1 = np.random.uniform(0.,50.,nsamples)
    #x = np.linspace(0., 50., nsamples)
    y = test_func(x1)
    noise = np.random.normal(0.,0.1,nsamples)
    y = y + noise

    # fix, ax = plt.subplots()
    # ax.plot(x1,y,'.')
    # plt.show()

    xv1 = np.abs(np.random.uniform(0.,50.,nsamples//100))
    yv = test_func(xv1)

    x = np.column_stack([x1, np.sqrt(x1), np.sin(x1), np.cos(x1), np.exp(-x1)])
    xv = np.column_stack([xv1, np.sqrt(xv1), np.sin(xv1), np.cos(xv1), np.exp(-xv1)])

    xt = torch.from_numpy(x).float()
    yt = torch.from_numpy(y.reshape((nsamples, 1))).float()

    xvt = torch.from_numpy(xv).float()
    yvt = torch.from_numpy(yv.reshape((nsamples//100, 1))).float()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    model = TestNet(5,80,70,60,50,30,20).to(device)
    # H1,H2,H3,H4,H5 = 80,70,50,30,20
    # model = torch.nn.Sequential(
    #             torch.nn.Linear(1, H1),
    #             torch.nn.ReLU(),
    #             torch.nn.Linear(H1, H2),
    #             torch.nn.ReLU(),
    #             torch.nn.Linear(H2, H3),
    #             torch.nn.ReLU(),
    #             torch.nn.Linear(H3, H4),
    #             torch.nn.ReLU(),
    #             torch.nn.Linear(H4, H5),
    #             torch.nn.ReLU(),
    #             torch.nn.Linear(H5, 1)
    #             ).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    x, y = xt.to(device), yt.to(device)
    xv, yv = xvt.to(device), yvt.to(device)
    n_epochs = 1000

    for epoch in range(n_epochs):
        ### Training step
        model.train()
        optimizer.zero_grad()

        y_ = model(x)
        loss = criterion(y_, y)

        print('epoch: ', epoch,' loss: ', loss.item())
        loss.backward()
        optimizer.step()

    ### Eval
    model.eval()
    with torch.no_grad():
        y_ = model(xv)
        loss = criterion(y_, yv)

    fig, ax = plt.subplots()
    ax.plot(xv1, y_.cpu().numpy(), '.', label='pred')
    ax.plot(xv1, yv.cpu().numpy(), '.', label='data')
    ax.set_title(f"MSE: {loss.item():0.5f}")
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()