from webcam_dataset import WebcamData
from webcam_dataset import Train
from webcam_dataset import Test
from webcam_dataset import Validation
from torchvision import transforms

data = WebcamData()

# TO DO: transforms

train = Train(data)
test = Test(data)
valid = Valid(data)


# Pickle or whatever all day objs?
# Does it make sense to save these patches to disk?

# Converting Pytorch from Numpy doesn't require extra memory
# b = torch.from_numpy(a)
## ToTensor() transform will do this

# How to take the day_obj and split into train, dev, test?
## Hashing

# Use PIL (Python Image Library) instead of CV2? RGB vs BGR
## Probably not, we want to stack image s anyway

# torchvision.transforms.CenterCrop or RandomCrop will grab patch FROM A PIL IMAGE

# Can write your own transforms http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# Could use extract_patches_2d

# Can write your own sampler to get train/test/valid http://pytorch.org/docs/master/_modules/torch/utils/data/sampler.html
# Based on the fact that here they use RandomSampler https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
# num_workers=1, pin-memory = true for GPU?  (see github link above)

# Could parallelize the load_images() in webcam_dataset