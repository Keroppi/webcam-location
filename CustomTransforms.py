import constants, torch, numpy as np
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
        img_stack = [0] * constants.IMAGES_PER_DAY
        for i in range(constants.IMAGES_PER_DAY):
            img = sample[i, :, :, :]
            # cv2.imwrite('/home/vli/patches/test' + str(int(i / constants.NUM_CHANNELS)) + '.jpg', img)

            img_stack[i] = extract_patches_2d(img, self.output_size, 1)[0]
            # cv2.imwrite('/home/vli/patches/test' + str(i) + '.jpg', patch)
        img_stack = np.stack(img_stack, axis=0)

        return img_stack

class ToTensor():
    def __call__(self, sample):
        num_images, height, width, num_channels = sample.shape
        reshaped = sample.reshape(constants.NUM_CHANNELS * constants.IMAGES_PER_DAY, height, width)
        #print(np.shares_memory(reshaped, sample))
        #print(reshaped.shape)

        return torch.from_numpy(reshaped)
