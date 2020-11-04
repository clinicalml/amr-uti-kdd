import torch
import torch.nn as nn
import torch.nn.functional as F

# Model architectures
class PolicyNetLinear(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(PolicyNetLinear, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_outputs)


    def forward(self, x, avoid_last=False):
        x = self.fc1(x)
        if avoid_last:
            x = x[:, :-1]

        return torch.softmax(x, dim=-1)


class PolicyNetSingleLayer(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(PolicyNetSingleLayer, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 256)
        self.fc2 = nn.Linear(256, num_outputs)

    def forward(self, x, avoid_last=False):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        if avoid_last:
            x = x[:, :-1]
        return torch.softmax(x, dim=-1)

# Loss functions
def policy_loss(output, target, avoid_last=False):
    if avoid_last:
        target = target[:, :-1]
         
    return -1*torch.mean(torch.log(output) * target)

def policy_loss_non_convex(output, target):
    return -1*torch.mean(output * target)
