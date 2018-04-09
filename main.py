import constants
import torch, torchvision
from webcam_dataset import WebcamData
from webcam_dataset import Train
from webcam_dataset import Test
from webcam_dataset import Validation
from CustomTransforms import RandomPatch, ToTensor


data = WebcamData()

transformations = torchvision.transforms.Compose([RandomPatch(constants.PATCH_SIZE), ToTensor()])

train_dataset = Train(data, transformations)
test_dataset = Test(data, transformations)
valid_dataset = Validation(data, transformations)

'''
print(len(train_dataset))
print(len(test_dataset))
print(len(valid_dataset))
for i in range(7):
    print(data.days[i].train_test_valid)
'''

if torch.cuda.is_available():
    pin_memory = True
    num_workers = 1
else:
    pin_memory = False
    num_workers = constants.NUM_LOADER_WORKERS

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=constants.BATCH_SIZE, num_workers=num_workers, pin_memory=pin_memory)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=constants.BATCH_SIZE, num_workers=num_workers, pin_memory=pin_memory)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=constants.BATCH_SIZE, num_workers=num_workers, pin_memory=pin_memory)


for e in range(constants.EPOCHS):
    loss = 0

    for i_batch, sample_batched in enumerate(train_loader):
        print(i_batch) # Batch idx
        print(sample_batched[0].size()) # Inputs
        print(sample_batched[1]) # Outputs




# Pickle or whatever all patches?
# Does it make sense to save these patches to disk?

# Could parallelize the load_images() in webcam_dataset
## No need for now since it's just strings.

# 3D Convolution

# Each img_stack = 32 * 3 * 128 * 128 bytes = 1.57 MB

# Could use GPU to transform images...?  ToTensor first step - https://discuss.pytorch.org/t/preprocess-images-on-gpu/5096