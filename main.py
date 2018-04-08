import constants
from webcam_dataset import WebcamData
from webcam_dataset import Train
from webcam_dataset import Test
from webcam_dataset import Validation
from CustomTransforms import RandomPatch, ToTensor
from torchvision import transforms

data = WebcamData()

transformations = transforms.Compose([RandomPatch(constants.PATCH_SIZE), ToTensor()])

train_dataset = Train(data, transformations)
test_dataset = Test(data, transformations)
valid_dataset = Validation(data, transformations)

train_dataset[0]

# Pickle or whatever all patches?
# Does it make sense to save these patches to disk?

# num_workers=1, pin_memory = true for GPU?  (see github link above) https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb

# Could parallelize the load_images() in webcam_dataset
## No need for now since it's just strings.

# 3D Convolution