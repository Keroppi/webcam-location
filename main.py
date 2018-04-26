#!/srv/glusterfs/vli/.pyenv/shims/python

import torch, torchvision, os, time, shutil, os, argparse, subprocess, sys, pickle

sys.path.append('/home/vli/webcam-location') # For importing .py files in the same directory on the cluster.
import constants

from webcam_dataset import WebcamData
from webcam_dataset import Train
from webcam_dataset import Test
from webcam_dataset import Validation
from custom_transforms import Resize, RandomResize, RandomPatch, ToTensor
from custom_model import WebcamLocation, RandomizeArgs
from torch.autograd import Variable

if constants.CLUSTER:
    d = dict(os.environ)
    print('SGE_GPU: ' + d['SGE_GPU'])
    sys.stdout.flush()

#print('Current Device(s): ' + str(torch.cuda.current_device()))
print('Device Count: ' + str(torch.cuda.device_count()))
sys.stdout.flush()

parser = argparse.ArgumentParser(description='Webcam Locator')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--load_model_args', default='', type=str, metavar='PATH',
                    help='path to pickled model args (default: none)')
args = parser.parse_args()

if not constants.CLUSTER:
    torch.backends.cudnn.enabled = False

data = WebcamData()

transformations = torchvision.transforms.Compose([RandomResize(constants.PATCH_SIZE), RandomPatch(constants.PATCH_SIZE), ToTensor()])
test_transformations = torchvision.transforms.Compose([Resize(), RandomPatch(constants.PATCH_SIZE), ToTensor()])

train_dataset = Train(data, transformations)
test_dataset = Test(data, test_transformations)
#valid_dataset = Validation(data, test_transformations)

if torch.cuda.is_available():
    pin_memory = True
else:
    print('WARNING - Not using GPU.')
    pin_memory = False

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=constants.BATCH_SIZE, num_workers=constants.NUM_LOADER_WORKERS, pin_memory=pin_memory)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=constants.BATCH_SIZE, num_workers=constants.NUM_LOADER_WORKERS, pin_memory=pin_memory)
#valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=constants.BATCH_SIZE, num_workers=num_workers, pin_memory=pin_memory)

#'''
print("Train/Test Sizes: ")
print(len(train_loader))
print(len(test_loader))
#print(len(valid_loader))
print(len(train_loader.dataset))
print(len(test_loader.dataset))
#print(len(valid_loader.dataset))
sys.stdout.flush()
#'''

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if not args.load_model_args:
    model_t0 = time.time()
    while True: # Try random models until we get one where the convolutions produce a valid size.
        try:
            model_args = RandomizeArgs()
            model = WebcamLocation(*model_args)
            model_memory_mb = count_parameters(model) * 4 / 1000 / 1000

            if constants.CLUSTER:
                if model_memory_mb < 2000: # Only proceed if the model's memory is less than 2 GB
                    print('Model memory (MB): ' + str(model_memory_mb))
                    sys.stdout.flush()

                    break
            else:
                if model_memory_mb < 200:
                    print('Model memory (MB): ' + str(model_memory_mb))
                    sys.stdout.flush()

                    break
        except RuntimeError as e: # Very hacky.
            if str(e).find('Output size is too small') >= 0: # Invalid configuration.
                pass
            elif str(e).find('not enough memory: you tried to allocate') >= 0: # Configuration uses too much memory.
                pass
            elif str(e).find("Kernel size can't greater than actual input size") >= 0: # Kernel is bigger than input.
                pass
            else:
                raise e

    model_t1 = time.time()
    print('Time to find a valid model (s): ' + str(model_t1 - model_t0))
    sys.stdout.flush()
else:
    with open(args.load_model_args, 'rb') as pkl_f:
        model_args = pickle.load(pkl_f)
        model = WebcamLocation(*model_args)

train_loss_fn = torch.nn.MSELoss()
test_loss_fn = torch.nn.MSELoss(size_average=False)

if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    #model = model.cuda()
    model.cuda()
    train_loss_fn = train_loss_fn.cuda() # Probably does nothing.
    test_loss_fn = test_loss_fn.cuda() # Probably does nothing.

optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-3)
start_epoch = 0
best_error = float('inf')

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model = checkpoint['model']
        start_epoch = checkpoint['epoch']
        best_error = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
        sys.stdout.flush()
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        sys.stdout.flush()
        sys.exit()

def train_epoch(epoch, model, data_loader, optimizer):
    model.train()

    for batch_idx, (data, target) in enumerate(data_loader):
        batch_train_time_t0 = time.time()
        data, target = Variable(data), Variable(target)

        target = target.float()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        if batch_idx == 0:
            vmem = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
            print('V-Memory Before Train Forward: ' + str(epoch) + '\n' + str(vmem.stdout).replace('\\n', '\n'))
            sys.stdout.flush()


        optimizer.zero_grad()
        output = model(data)
        loss = train_loss_fn(output, target)
        loss.backward()

        if batch_idx == 0:
            vmem = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
            print('V-Memory After Train Backward: ' + str(epoch) + '\n' + str(vmem.stdout).replace('\\n', '\n'))
            sys.stdout.flush()

        optimizer.step()

        batch_train_time_t1 = time.time()
        batch_train_time_min = (batch_train_time_t1 - batch_train_time_t0) / 60

        if batch_idx % constants.LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] | Batch Loss: {:.4f} | Total Time (min): {:.4f}'.format(
                  epoch, batch_idx * len(data), len(data_loader.dataset),
                  100. * batch_idx / len(data_loader), loss.data[0], batch_train_time_min))
            sys.stdout.flush()

        del loss  # https://discuss.pytorch.org/t/best-practices-for-maximum-gpu-utilization/13863/5



def test_epoch(model, data_loader):
    model.eval()
    test_loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = Variable(data, volatile=True), Variable(target)
        target = target.float()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        output = model(data)
        test_loss += test_loss_fn(output, target).data[0] # sum up batch loss

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

    vmem = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    print('V-Memory After Test: \n' + str(vmem.stdout).replace('\\n', '\n'))

    return test_loss

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if constants.CLUSTER:
        directory = '/srv/glusterfs/vli/models/'
    else:
        directory = '~/models/'
        directory = os.path.expanduser(directory)

    if constants.LEARNING_SUNRISE:
        prefix = 'sunrise_'
    else:
        prefix = 'sunset_'

    torch.save(state, directory + prefix + filename)
    if is_best:
        shutil.copyfile(directory + prefix + filename, directory + prefix + 'model_best.pth.tar')
        with open(directory + prefix + 'best_params.txt', 'w') as best_params:
            best_params.write(str(model.features) + '\n')
            best_params.write(str(model.regressor) + '\n')
            #best_params.write('Max Pooling: ' + str(model_args[5]) + '\n')
            #best_params.write('Conv Relus: ' + str(model_args[6]) + '\n')
            #best_params.write('FC Relus: ' + str(model_args[9]) + '\n')
            best_params.write('Using ' + str(constants.DAYS_PER_MONTH) + ' days per month.\n')
            best_params.write('Using ' + str(state['epoch']) + ' epochs.\n')
            best_params.write('Using batch size of ' + str(constants.BATCH_SIZE) + '.\n')
            best_params.write(str(state['best_prec1']))


for epoch in range(start_epoch, constants.EPOCHS):
    print('Epoch: ' + str(epoch))
    sys.stdout.flush()

    train_t0 = time.time()
    train_epoch(epoch, model, train_loader, optimizer)
    train_t1 = time.time()
    print('Epoch Train Time (min): ' + str((train_t1 - train_t0) / 60))
    sys.stdout.flush()

    test_t0 = time.time()
    test_error = test_epoch(model, test_loader)
    test_t1 = time.time()
    print('Epoch Test Time (min): ' + str((test_t1 - test_t0) / 60))
    sys.stdout.flush()

    is_best = test_error < best_error
    best_error = min(test_error, best_error)
    save_checkpoint({
        'model': model,
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_error,
        'optimizer': optimizer.state_dict(),
    }, is_best)



# Pickle or whatever all patches?
# Does it make sense to save these patches to disk?

# Could use GPU to transform images...?  ToTensor first step - https://discuss.pytorch.org/t/preprocess-images-on-gpu/5096

# Multiple GPUs
# http://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html