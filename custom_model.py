import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import constants

class WebcamLocation(nn.Module):
    def __init__(self):
        super(WebcamLocation, self).__init__()

        kernel_sizes = [(3, 3), (5, 5), (7, 7)] # each element corresponds to a layer, tuple is (height, width)

        self.conv1 = nn.Conv3d(constants.NUM_CHANNELS, 1, kernel_size=(1, kernel_sizes[0][0], kernel_sizes[0][1]), stride=1)
        self.conv2 = nn.Conv3d(3, 2, kernel_size=(1, kernel_sizes[1][0], kernel_sizes[1][1]), stride=1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(constants.FIRST_FC_LAYER_SIZE, 1)
        #self.fc2 = nn.Linear(100, 70)
        #self.fc3 = nn.Linear(70, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        #print(self.num_flat_features(x)) # 32 * 128 * 128 * 3
        x = self.conv1(x) # DELETE ME
        print(self.num_flat_features(x))
        #x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features