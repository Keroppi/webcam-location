import torch, sys
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import constants

class WebcamLocation(nn.Module):
    def __init__(self):
        super(WebcamLocation, self).__init__()

        kernel_sizes = [(1, 5, 5),
                        (4, 2, 2), # Look at 4 frames at once.
                        (1, 2, 2),
                        (1, 2, 2)] # each element corresponds to a layer, tuple is (height, width)
        output_channels = [16, 32, 48, 16] # each element corresponds to a layer
        padding = [(2, 2), (1, 1), (1, 1), (0, 0)] # each element corresponds to a layer

        linear_sizes = [2000, 200]


        self.conv1 = nn.Conv3d(constants.NUM_CHANNELS, output_channels[0],
                               kernel_size=kernel_sizes[0],
                               stride=(1, 2, 2),
                               padding=(0, padding[0][0], padding[0][1]))
        self.conv2 = nn.Conv3d(output_channels[0], output_channels[1],
                               kernel_size=kernel_sizes[1],
                               stride=(2, 1, 1), # Skip every other frame.
                               padding=(0, padding[1][0], padding[1][1]))
        self.conv3 = nn.Conv3d(output_channels[1], output_channels[2],
                               kernel_size=kernel_sizes[2],
                               stride=1,
                               padding=(0, padding[2][0], padding[2][1]))
        self.conv4 = nn.Conv3d(output_channels[2], output_channels[3],
                               kernel_size=kernel_sizes[3],
                               stride=1,
                               padding=(0, padding[3][0], padding[3][1]))

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(constants.FIRST_FC_LAYER_SIZE, linear_sizes[0])
        self.fc2 = nn.Linear(linear_sizes[0], linear_sizes[1])
        self.fc3 = nn.Linear(linear_sizes[1], 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool3d(F.relu(self.conv1(x)), (1, 2, 2))

        #print('Conv 1')
        #print(x.size())
        #print(self.num_flat_features(x))

        x = F.max_pool3d(F.relu(self.conv2(x)), (1, 2, 2))

        #print('Conv 2')
        #print(x.size())
        #print(self.num_flat_features(x))

        x = F.relu(self.conv3(x))

        #print('Conv 3')
        #print(x.size())
        #print(self.num_flat_features(x))

        x = F.max_pool3d(F.relu(self.conv4(x)), (1, 2, 2))

        print('Conv 4')
        print(x.size())
        print(self.num_flat_features(x))

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features