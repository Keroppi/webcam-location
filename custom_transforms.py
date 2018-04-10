import constants, torch, torchvision, numpy as np, cv2, PIL, sys, random
from sklearn.feature_extraction.image import extract_patches_2d

class RandomPatch():
    def __init__(self, output_size): # int for square, else (height, width)
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        random_int = random.randint(0, 2 ** 32 - 1)  # To consistently look at one patch (instead of random patches).

        img_stack = [0] * constants.IMAGES_PER_DAY
        for i in range(constants.IMAGES_PER_DAY):
            img = sample[i, :, :, :]
            # cv2.imwrite('/home/vli/patches/test' + str(int(i / constants.NUM_CHANNELS)) + '.jpg', img)

            img_stack[i] = extract_patches_2d(img, self.output_size, max_patches=1, random_state=random_int)[0]
            # cv2.imwrite('/home/vli/patches/test' + str(i) + '.jpg', patch)
            #patch = PIL.Image.fromarray(np.uint8(img_stack[i]))
            #patch.save('/home/vli/patches/sample' + str(i) + '.jpg')
        img_stack = np.stack(img_stack, axis=0)

        return img_stack

class ToTensor():
    def __call__(self, sample):
        num_images, height, width, num_channels = sample.shape
        transposed = sample.transpose(3, 0, 1, 2) # NUM_IMAGES_PER_DAY x C x H x W
        #reshaped = transposed.reshape(constants.IMAGES_PER_DAY * 3, height, width) # C x H x W

        #original = PIL.Image.fromarray(np.uint8(sample[5]))
        #original.save('/home/vli/patches/sample.jpg')

        #print(np.shares_memory(transposed, sample))
        #print(transposed.shape)

        torch_image = torch.from_numpy(transposed)
        torch_image = torch_image.float().div(255) # scale to [0, 1], and create float tensor
        #torch_image = torch.from_numpy(reshaped)

        #print(sample[5, :, :, 1])
        #print(torch_image.numpy()[16:17, :, :])

        #tensor = PIL.Image.fromarray(np.uint8(torch_image.numpy()[5, :, :, :].transpose(1, 2, 0)))
        #tensor.save('/home/vli/patches/torch.jpg')
        #torchvision.utils.save_image(torch_image[5], '/home/vli/patches/torch.jpg', padding=0, normalize=True, range=(0, 255))

        return torch_image
