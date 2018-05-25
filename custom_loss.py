import constants
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class CustomLoss(torch.nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(CustomLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, x, y):
        mse_part = F.mse_loss(x[:, 0], y[:, 0], reduce=False)
        cel_part = F.binary_cross_entropy(x[:, 1], y[:, 1], reduce=False)
        loss = mse_part * x[:, 1] + constants.LAMBDA + cel_part

        if not self.reduce:
            return loss

        total_loss = torch.sum(loss)

        if self.size_average:
            return total_loss / x.size()[0]
        else:
            return total_loss

'''
a = torch.ones([5, 2]) * 0.5 # sunrise or sunset idx, followed by a probability that the day is good
b = torch.ones([5, 2]) # sunrise or sunset idx, followed by 1 if day is green or 0 if black
a = Variable(a)
b = Variable(b)

cl = CustomLoss()
print(cl(a, b))
'''