import torch, torchvision, os, datetime, time
import constants
from webcam_dataset import WebcamData
from webcam_dataset import Test
from custom_transforms import Resize, RandomPatch, ToTensor
from torch.autograd import Variable

if constants.CLUSTER:
    directory = '/srv/glusterfs/vli/models/best/'
else:
    directory = '~/models/best/'
    directory = os.path.expanduser(directory)

sunrise_model = directory + 'sunrise_model_best.pth.tar'
sunset_model = directory + 'sunset_model_best.pth.tar'

sunrise_checkpt = torch.load(sunrise_model)
sunset_checkpt = torch.load(sunset_model)

sunrise_model = sunrise_checkpt['model']
sunset_model = sunset_checkpt['model']

sunrise_model.eval()
sunset_model.eval()

data = WebcamData()
test_transformations = torchvision.transforms.Compose([Resize(), RandomPatch(constants.PATCH_SIZE), ToTensor()])
test_dataset = Test(data, test_transformations)

if torch.cuda.is_available():
    pin_memory = True
    num_workers = 1
else:
    print('WARNING - Not using GPU.')
    pin_memory = False
    num_workers = constants.NUM_LOADER_WORKERS

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=constants.BATCH_SIZE, num_workers=num_workers, pin_memory=pin_memory)

utc_sunrises = []
for batch_idx, (input, _) in enumerate(test_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunrise_idx = sunrise_model(input)

    # Convert sunrise_idx into a local time.

    # Assume that data is in the same order as the batch, since it's not shuffled.
    batch_days = data[batch_idx * constants.BATCH_SIZE:(batch_idx + 1) * constants.BATCH_SIZE]

    for d_idx, day in enumerate(batch_days):
        local_sunrise = day.get_local_time(sunrise_idx[d_idx, 0])
        utc_sunrise = local_sunrise - datetime.timedelta(seconds=day.time_offset)
        utc_sunrises.append(utc_sunrise)

utc_sunsets = []
for batch_idx, (input, _) in enumerate(test_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunset_idx = sunset_model(input)

    # Convert sunset_idx into a local time.

    # Assume that data is in the same order as the batch, since it's not shuffled.
    batch_days = data[batch_idx * constants.BATCH_SIZE:(batch_idx + 1) * constants.BATCH_SIZE]

    for d_idx, day in enumerate(batch_days):
        local_sunset = day.get_local_time(sunrise_idx[d_idx, 0])
        utc_sunset = local_sunset - datetime.timedelta(seconds=day.time_offset)
        utc_sunsets.append(utc_sunset)

# Compute solar noon.



# Compute day length.



# Get solar noon and day length, compute lat/lng
# Calculate diff between that and actual