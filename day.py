import constants, numpy as np, cv2
from sklearn.feature_extraction.image import extract_patches_2d

class Day():
    def get_sun_idx(self, times, sunrise, sunset):
        sunrise_idx = 0
        sunset_idx = 0

        max_sunrise_idx = None
        max_sunset_idx = None
        for time in times:
            if sunrise < time:
                max_sunrise_idx = sunrise_idx
                break

            sunrise_idx += 1
            sunset_idx += 1

        for time in times:
            if sunset < time:
                max_sunset_idx = sunset_idx
                break

            sunrise_idx += 1
            sunset_idx +=1

        if max_sunrise_idx is None:
            sunrise_idx = constants.IMAGES_PER_DAY # Past the last image.
        elif max_sunrise_idx == 0:
            sunrise_idx = -1
        else:
            remainder = (sunrise - times[max_sunrise_idx - 1]) / (times[max_sunrise_idx] - times[max_sunrise_idx - 1])
            sunrise_idx = remainder + max_sunrise_idx - 1

        if max_sunset_idx is None:
            sunset_idx = constants.IMAGES_PER_DAY
        elif max_sunset_idx == 0:
            sunset_idx = -1
        else:
            remainder = (sunset - times[max_sunset_idx - 1]) / (times[max_sunset_idx] - times[max_sunset_idx - 1])
            sunset_idx = remainder + max_sunset_idx - 1

        return (sunrise_idx, sunset_idx)

    def __init__(self, times, img_stack, sunrise, sunset):
        self.date = times[0].date()
        #self.img_stack = img_stack
        self.sunrise = sunrise
        self.sunset = sunset

        # Randomly select IMAGES_PER_DAY images from times.
        subset_time_idx = np.random.choice(len(times), constants.IMAGES_PER_DAY, replace=False)
        subset_time_idx.sort()

        subset_times = [times[x] for x in subset_time_idx]
        self.sunrise_idx, self.sunset_idx = self.get_sun_idx(subset_times, sunrise, sunset)

        # Extract only the subset of images.
        subset_img_idx = []
        for idx in subset_time_idx:
            for channel in range(constants.NUM_CHANNELS):
                subset_img_idx += [constants.NUM_CHANNELS * idx + channel]

        subset_img_stack = np.take(img_stack, subset_img_idx, axis=2)

        #print(subset_time_idx)
        #print(subset_img_idx)

        # Cut out patches from the images.
        self.patch_stack = np.array([]) # Stack patches along the color channel depth.
        for i in range(0, constants.IMAGES_PER_DAY * constants.NUM_CHANNELS, constants.NUM_CHANNELS):
            img = subset_img_stack[:, :, i:i+constants.NUM_CHANNELS]
            #cv2.imwrite('/home/vli/patches/test' + str(-i) + '.jpg', img)

            patch = extract_patches_2d(img, (constants.PATCH_H, constants.PATCH_W), 1)[0]
            #cv2.imwrite('/home/vli/patches/test' + str(i) + '.jpg', patch)

            self.patch_stack = np.dstack((self.patch_stack, patch)) if self.patch_stack.size else patch

        












