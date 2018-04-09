import constants
import torch, torchvision, os, time
from webcam_dataset import WebcamData
from webcam_dataset import Train
from webcam_dataset import Test
from webcam_dataset import Validation
from custom_transforms import RandomPatch, ToTensor
from custom_model import WebcamLocation
from torch.autograd import Variable
import torch.nn.functional as F

if not constants.CLUSTER:
    torch.backends.cudnn.enabled = False

data = WebcamData()

transformations = torchvision.transforms.Compose([RandomPatch(constants.PATCH_SIZE), ToTensor()])

train_dataset = Train(data, transformations)
test_dataset = Test(data, transformations)
valid_dataset = Validation(data, transformations)

if torch.cuda.is_available():
    pin_memory = True
    num_workers = 1
else:
    pin_memory = False
    num_workers = constants.NUM_LOADER_WORKERS

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=constants.BATCH_SIZE, num_workers=num_workers, pin_memory=pin_memory)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=constants.BATCH_SIZE, num_workers=num_workers, pin_memory=pin_memory)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=constants.BATCH_SIZE, num_workers=num_workers, pin_memory=pin_memory)

model = WebcamLocation()
model.cuda()

loss_fn = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-4)

for e in range(constants.EPOCHS):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()

        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()

        output = model(data)

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

    #print(i_batch) # Batch idx
    #print(sample_batched[0].size()) # Inputs
    #print(sample_batched[1]) # Outputs

# FOLLOW THIS FOR NOW https://github.com/pytorch/examples/blob/master/mnist_hogwild/train.py

# Pickle or whatever all patches?
# Does it make sense to save these patches to disk?

# Could parallelize the load_images() in webcam_dataset
## No need for now since it's just strings.

# 3D Convolution

# Each img_stack = 32 * 3 * 128 * 128 bytes = 1.57 MB

# Could use GPU to transform images...?  ToTensor first step - https://discuss.pytorch.org/t/preprocess-images-on-gpu/5096

# Scale images to [0, 1] ?