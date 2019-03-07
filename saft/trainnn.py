from torch import nn
from torch import optim

class TestNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.