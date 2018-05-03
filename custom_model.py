import torch, sys, random, pickle
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import constants

class WebcamLocation(nn.Module):
    def __init__(self,
                 num_conv_layers, output_channels, kernel_sizes, paddings, strides, max_poolings, max_pooling_strides, conv_relus, # Conv inputs.
                 num_hidden_fc_layers, fc_sizes, fc_relus, # FC inputs.
                 input_shape=(3, 32, 128, 128)):
        super(WebcamLocation, self).__init__()

        ### Convolutional layers ###

        self.num_conv_layers = num_conv_layers
        self.kernel_sizes = kernel_sizes
        self.output_channels = output_channels
        self.paddings = paddings
        self.strides = strides
        self.num_pool_layers = len(max_poolings) - max_poolings.count(None)
        self.max_poolings = max_poolings
        self.max_pooling_strides = max_pooling_strides
        self.num_conv_relu_layers = conv_relus.count(True)
        self.conv_relus = conv_relus
        self.conv_layers = [0] * (self.num_conv_layers + self.num_pool_layers + self.num_conv_relu_layers) # Initialize

        # Example below for 4 CNN layers:
        '''
        self.kernel_sizes = [(1, 5, 5),
                             (4, 2, 2), # Look at 4 frames at once.
                             (1, 2, 2),
                             (1, 2, 2)] # each element corresponds to a layer, tuple is (height, width)
        self.output_channels = [16, 32, 48, 16] # each element corresponds to a layer
        self.paddings = [(0, 2, 2), (0, 1, 1), (0, 1, 1), (0, 0, 0)] 
        self.strides = [(1, 2, 2), (2, 1, 1), (1, 1, 1), (1, 1, 1)]
        self.poolings = [(1, 2, 2), (1, 2, 2), None, (1, 2, 2)]
        self.conv_relus = [True, True, False, True]
        '''

        j = 0
        for i in range(self.num_conv_layers):
            if i == 0:
                self.conv_layers[j] = nn.Conv3d(constants.NUM_CHANNELS,
                                                self.output_channels[i],
                                                kernel_size=self.kernel_sizes[i],
                                                stride=self.strides[i],
                                                padding=self.paddings[i])
            else:
                self.conv_layers[j] = nn.Conv3d(self.output_channels[i - 1],
                                                self.output_channels[i],
                                                kernel_size=self.kernel_sizes[i],
                                                stride=self.strides[i],
                                                padding=self.paddings[i])
            j += 1

            if self.conv_relus[i]:
                self.conv_layers[j] = nn.ReLU()
                j += 1
            if self.max_poolings[i] is not None:
                self.conv_layers[j] = nn.MaxPool3d(self.max_poolings[i], self.max_pooling_strides[i])
                j += 1

        # Example below for 4 CNN layers:
        '''
        self.conv1 = nn.Conv3d(constants.NUM_CHANNELS, self.output_channels[0],
                               kernel_size=self.kernel_sizes[0],
                               stride=self.strides[0],
                               padding=self.paddings[0])
        self.conv2 = nn.Conv3d(self.output_channels[0], self.output_channels[1],
                               kernel_size=self.kernel_sizes[1],
                               stride=self.strides[1], # Skip every other frame.
                               padding=self.paddings[1])
        self.conv3 = nn.Conv3d(self.output_channels[1], self.output_channels[2],
                               kernel_size=self.kernel_sizes[2],
                               stride=self.strides[2],
                               padding=self.paddings[2])
        self.conv4 = nn.Conv3d(self.output_channels[2], self.output_channels[3],
                               kernel_size=self.kernel_sizes[3],
                               stride=self.strides[3],
                               padding=self.paddings[3])
        '''

        self.features = nn.Sequential(*self.conv_layers)

        ### Fully connected layers ###

        self.num_hidden_fc_layers = num_hidden_fc_layers
        self.fc_sizes = fc_sizes
        self.fc_relus = fc_relus + [False] # Output layer doesn't need ReLu
        self.fc_layers = [0] * (self.num_hidden_fc_layers + 1 + self.fc_relus.count(True))

        # Compute output size of convolutions to get input to fc layers.
        self.first_fc_layer_size = self.get_conv_output(input_shape)

        # an affine operation: y = Wx + b
        j = 0
        for i in range(self.num_hidden_fc_layers + 1):
            if i == 0:
                self.fc_layers[j] = nn.Linear(self.first_fc_layer_size, self.fc_sizes[0])
            elif i < self.num_hidden_fc_layers:
                self.fc_layers[j] = nn.Linear(self.fc_sizes[i - 1], self.fc_sizes[i])
            else:
                self.fc_layers[j] = nn.Linear(self.fc_sizes[i - 1], 1)

            j += 1

            if self.fc_relus[i]:
                self.fc_layers[j] = nn.ReLU()
                j += 1


        # Example below for 2 hidden FC layers:
        '''
        # Example (2 hidden layers):
        linear_sizes = [2000, 200]
        self.fc1 = nn.Linear(first_fc_layer_size, linear_sizes[0])
        self.fc2 = nn.Linear(linear_sizes[0], linear_sizes[1])
        self.fc3 = nn.Linear(linear_sizes[1], 1)
        '''

        #self.network = nn.Sequential(*(self.conv_layers + self.fc_layers))


        self.regressor = nn.Sequential(*self.fc_layers)
        #print(self.features)
        #print(self.regressor)
        #sys.stdout.flush()


    # Used to get output size of convolutions.
    def get_conv_output(self, shape):
        self.features.eval()
        batch_size = 1 # Not important.
        input = Variable(torch.rand(batch_size, *shape), requires_grad=False)
        #print('Is this random variable cuda? ' + str(input.is_cuda)) # False
        #sys.stdout.flush()
        output_feat = self.features(input)
        self.features.train()
        return self.num_flat_features(output_feat)

    #def forward_features(self, x):
    #    for i in range(self.num_conv_layers):
    #        x = self.conv_layers[i](x)
    #        if self.conv_relus[i]:
    #            x = F.relu(x)

    #       if self.max_poolings[i] is not None:
    #            x = F.max_pool3d(x, self.max_poolings[i])

    #    return x

    def forward(self, x):
        # Convolutional layers.
        #x = self.forward_features(x)
        x = self.features(x)

        '''
        # Max pooling over a (2, 2) window
        x = F.max_pool3d(F.relu(self.conv1(x)), (1, 2, 2))
        x = F.max_pool3d(F.relu(self.conv2(x)), (1, 2, 2))
        x = F.relu(self.conv3(x))
        x = F.max_pool3d(F.relu(self.conv4(x)), (1, 2, 2))

        print('Conv 4')
        print(x.size())
        print(self.num_flat_features(x))
        '''

        # Flatten into vector.
        x = x.view(-1, self.first_fc_layer_size)

        # Fully connected layers.
        #for i in range(self.num_hidden_fc_layers + 1):
        #    x = self.fc_layers[i](x)

        #    if self.fc_relus[i]:
        #        x = F.relu(x)

        '''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        '''

        return self.regressor(x)


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features