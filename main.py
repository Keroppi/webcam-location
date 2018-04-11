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

print("Train / Test/ Validation Sizes: ")
print(len(train_loader))
print(len(test_loader))
print(len(valid_loader))

model = WebcamLocation()
model.cuda()

train_loss_fn = torch.nn.MSELoss().cuda()
test_loss_fn = torch.nn.MSELoss(size_average=False).cuda()
optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-3)

def train_epoch(epoch, model, data_loader, optimizer):
    model.train()

    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.cuda()
        target = target.float().cuda()

        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()

        output = model(data)

        loss = train_loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % constants.LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(data), len(data_loader.dataset),
                  100. * batch_idx / len(data_loader), loss.data[0]))

def test_epoch(model, data_loader):
    model.eval()
    test_loss = 0
    for data, target in data_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.cuda()
        target = target.float().cuda()

        output = model(data)
        test_loss += test_loss_fn(output, target).data[0] # sum up batch loss

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(
          test_loss))


for epoch in range(constants.EPOCHS):
    print('Epoch: ' + str(epoch))

    train_epoch(epoch, model, train_loader, optimizer)
    test_epoch(model, test_loader)



# Pickle or whatever all patches?
# Does it make sense to save these patches to disk?

# Could parallelize the load_images() in webcam_dataset
## No need for now since it's just strings.

# Each img_stack = 32 * 3 * 128 * 128 bytes = 1.57 MB

# Could use GPU to transform images...?  ToTensor first step - https://discuss.pytorch.org/t/preprocess-images-on-gpu/5096

# Could just artificially limit # of days in this line below...
# self.data = data.days[num_test:num_test + num_train]
# change __len__ function as well

# Ycbcr
## https://stackoverflow.com/questions/24610775/pil-image-convert-from-rgb-to-ycbcr-results-in-4-channels-instead-of-3-and-behav