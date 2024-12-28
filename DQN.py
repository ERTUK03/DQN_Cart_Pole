import torch

class DQN(torch.nn.Module):
  def __init__(self, inputs, outputs):
    super(DQN, self).__init__()
    self.layer1 = torch.nn.Linear(inputs, 128)
    self.layer2 = torch.nn.Linear(128, 128)
    self.layer3 = torch.nn.Linear(128, outputs)
    self.relu = torch.nn.ReLU()

  def forward(self, x):
    x = self.relu(self.layer1(x))
    x = self.relu(self.layer2(x))
    return self.layer3(x)
