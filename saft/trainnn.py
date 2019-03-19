import torch
import numpy as np 
import pandas as pd 
from torch import nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt 

from sklearn.datasets import make_regression

class TestNet(nn.Module):
    def __init__(self, input_output, *args):
        super().__init__()
        input_dim = input_output[0]
        output_dim = input_output[1]
        self.beta = nn.ModuleList()
        self.sig = nn.Tanh()
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

    # Set up data into numpy
    df = pd.read_csv('PvTdata.csv', header=None)
    df = df[df.iloc[:,4] <= 30]
    k = 1.38064852e-23
    Na = 6.02214086e23
    v = df.iloc[:,0].values
    rho = df.iloc[:,1].values
    T = df.iloc[:,2].values
    p_eos = df.iloc[:,3].values
    p_real = df.iloc[:,4].values
    z = p_real * 1e5 / (Na * k * rho * T)

    nsamples = 50000
    N_S = len(v)
    x1 = np.random.uniform(0.,50.,nsamples)
    #x = np.linspace(0., 50., nsamples)
    y = test_func(x1)
    noise = np.random.normal(0.,0.1,nsamples)
    y = y + noise

    # fix, ax = plt.subplots()
    # ax.plot(x1,y,'.')
    # plt.show()

    # Splitting validation and training samples
    xv1 = np.abs(np.random.uniform(0.,50.,nsamples//100))
    yv = test_func(xv1)

    # Manipulation of input 
    x = np.column_stack([x1])
    xv = np.column_stack([xv1])
    PvTdata = np.column_stack([np.log(v), T, p_real, z])
    PvT = np.copy(PvTdata)
    np.random.shuffle(PvTdata)
    vTdata = PvTdata[:,0:2]
    p_real = PvTdata[:,2:3]
    z = PvTdata[:,3:4]

    # Convert to torch variables
    xt = torch.from_numpy(x).float()
    yt = torch.from_numpy(y.reshape((nsamples, 1))).float()

    xvt = torch.from_numpy(PvT[:,0:2]).float()
    yvt = torch.from_numpy(PvT[:,3:4]).float()

    vt = torch.from_numpy(vTdata).float()
    pt = torch.from_numpy(z).float()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())

    # Set up model, optimiser and loss function
    model = TestNet((2,1), *[200,100],20).to(device)
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

    # Load data
    x, y = xt.to(device), yt.to(device)
    xv, yv = xvt.to(device), yvt.to(device)

    x, y = vt.to(device), pt.to(device)
    n_epochs = 100

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

    # ax.plot(xv.cpu().numpy(), y_.cpu().numpy(), '.', label='pred')
    # ax.plot(xv.cpu().numpy(), yv.cpu().numpy(), '.', label='data')
    
    ax.plot(xv.cpu().numpy()[0:1000,0], y_.cpu().numpy()[0:1000], '.', label='pred')
    ax.plot(xv.cpu().numpy()[0:1000,0], yv.cpu().numpy()[0:1000], '.', label='data')
    ax.plot(xv.cpu().numpy()[100000:101000,0], y_.cpu().numpy()[100000:101000], '.', label='pred')
    ax.plot(xv.cpu().numpy()[100000:101000,0], yv.cpu().numpy()[100000:101000], '.', label='data')
    # ax.plot(PvTdata[180000:181000,0], y_.cpu().numpy()[180000:181000], '.', label='pred')
    # ax.plot(PvT[180000:181000,0], yv.cpu().numpy()[180000:181000], '.', label='data')
    ax.set_title(f"MSE: {loss.item():0.5f}")
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()