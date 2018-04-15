import constants
import numpy as np, sys, PIL
from sklearn.feature_extraction.image import extract_patches_2d

class Day():
    def get_sun_idx(self, times, sunrise, sunset):
        sunrise_idx = 0
        sunset_idx = 0

        # Time between first and last images.
        #Uused for scaling if the sunrise / sunset fall outside the range of images.
        time_diff = times[-1] - times[0]

        max_sunrise_idx = None
        max_sunset_idx = None
        for time in times:
            if sunrise < time:
                max_sunrise_idx = sunrise_idx
                break

            sunrise_idx += 1

        for time in times:
            if sunset < time:
                max_sunset_idx = sunset_idx
                break

            sunset_idx +=1

        if max_sunrise_idx is None: # Past the last image.
            sunrise_idx = constants.IMAGES_PER_DAY - 1
            extra = (sunrise - times[-1]) / time_diff
            sunrise_idx += extra
        elif max_sunrise_idx == 0: # Before the first image.
            sunrise_idx = 0
            extra = (sunrise - times[0]) / time_diff
            sunrise_idx += extra
        else:
            remainder = (sunrise - times[max_sunrise_idx - 1]) / (times[max_sunrise_idx] - times[max_sunrise_idx - 1])
            sunrise_idx = remainder + max_sunrise_idx - 1

        if max_sunset_idx is None:
            sunset_idx = constants.IMAGES_PER_DAY - 1
            extra = (sunset - times[-1]) / time_diff
            sunset_idx += extra
        elif max_sunset_idx == 0:
            sunset_idx = 0
            extra = (sunset- times[0]) / time_diff
            sunset_idx += extra
        else:
            remainder = (sunset - times[max_sunset_idx - 1]) / (times[max_sunset_idx] - times[max_sunset_idx - 1])
            sunset_idx = remainder + max_sunset_idx - 1

        #print(sunrise_idx)
        #print(sunrise)
        #print(times[max_sunrise_idx - 1])
        #print(times[max_sunrise_idx])

        #print(sunset_idx)
        #print(sunset)
        #print(times[max_sunset_idx - 1])
        #print(times[max_sunset_idx])
        #print('')

        return (sunrise_idx, sunset_idx)

    def __init__(self, times, img_paths, sunrise, sunset, train_test_valid):
        self.train_test_valid = train_test_valid
        self.date = times[0].date()
        self.img_paths = img_paths

        # Determine height / width
        # Perhaps unnecessary?
        example_img = PIL.Image.open(img_paths[0]) # lazily loads - the entire image is not read into memory
        self.width, self.height = example_img.size

        self.sunrise = sunrise
        self.sunset = sunset
        self.sunrise_idx, self.sunset_idx = self.get_sun_idx(times, sunrise, sunset)











