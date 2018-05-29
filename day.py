import constants
import numpy as np, sys, PIL, math, time, datetime, random

class Day():
    def get_local_time(self, idx):
        if idx <= constants.IMAGES_PER_DAY - 1 and idx >= 0:
            floor_idx = math.floor(idx)
            ceil_idx = math.ceil(idx)

            if floor_idx == ceil_idx:
                return self.times[floor_idx]

            diff = self.times[ceil_idx] - self.times[floor_idx]
            return self.times[floor_idx] + (idx - floor_idx) * diff
        elif idx < 0:
            diff = self.times[-1] - self.times[0]
            return self.times[0] + idx * diff
        else:
            diff = self.times[-1] - self.times[0]
            return self.times[constants.IMAGES_PER_DAY - 1] + (idx - (constants.IMAGES_PER_DAY - 1)) * diff

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

            sunset_idx += 1

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
            extra = (sunset - times[0]) / time_diff
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


    def random_subset(all_times, all_img_paths):
        # Randomly select IMAGES_PER_DAY images from times / images.
        subset_idx = np.random.choice(len(all_times), constants.IMAGES_PER_DAY, replace=False)
        subset_idx.sort()
        times = [all_times[x] for x in subset_idx]
        img_paths = [all_img_paths[x] for x in subset_idx]

        return (times, img_paths)

    def change_frames(self, center_frame): # Given a suggested frame idx, repick frames that are close to it.
        if center_frame < 0:
            start = 0
            end = 1
        elif center_frame >= constants.IMAGES_PER_DAY - 1:
            start = constants.IMAGES_PER_DAY - 2
            end = constants.IMAGES_PER_DAY - 1
        else:
            start = math.floor(center_frame)
            end = math.ceil(center_frame)

        start_pivot_time = self.times[start]
        end_pivot_time = self.times[end]

        for t_idx, time in enumerate(self.all_times):
            if time == start_pivot_time:
                start_idx = max(t_idx - 1, 0)
            elif time == end_pivot_time:
                end_idx = min(t_idx + 1, len(self.all_times) - 1)

        num_important_frames = end_idx - start_idx + 1
        if num_important_frames > constants.IMAGES_PER_DAY:
            end_idx = start_idx + constants.IMAGES_PER_DAY - 1

        important_frames = list(range(start_idx, end_idx + 1))
        remaining = set(range(len(self.all_times))) - set(important_frames)

        subset_idx = np.random.choice(len(remaining), constants.IMAGES_PER_DAY - len(important_frames), replace=False)
        subset_idx.sort()
        subset_idx = [list(remaining)[x] for x in subset_idx]
        subset_idx = subset_idx + important_frames
        subset_idx.sort()

        '''
        print(center_frame)
        print(self.times)
        print(self.all_times)
        print(subset_idx)
        print(len(subset_idx))
        print('')
        sys.stdout.flush()
        '''

        self.times = [self.all_times[x] for x in subset_idx]
        self.img_paths = [self.all_img_paths[x] for x in subset_idx]

        self.sunrise_idx, self.sunset_idx = self.get_sun_idx(self.times, self.sunrise, self.sunset)

        if self.sunrise_idx >= 0 and self.sunrise_idx <= constants.IMAGES_PER_DAY - 1:
            self.sunrise_in_frames = True
        else:
            self.sunrise_in_frames = False

        if self.sunset_idx >= 0 and self.sunset_idx <= constants.IMAGES_PER_DAY - 1:
            self.sunset_in_frames = True
        else:
            self.sunset_in_frames = False

    def random_frames(self):
        self.times, self.img_paths = Day.random_subset(self.all_times, self.all_img_paths)

        self.sunrise_idx, self.sunset_idx = self.get_sun_idx(self.times, self.sunrise, self.sunset)

        if self.sunrise_idx >= 0 and self.sunrise_idx <= constants.IMAGES_PER_DAY - 1:
            self.sunrise_in_frames = True
        else:
            self.sunrise_in_frames = False

        if self.sunset_idx >= 0 and self.sunset_idx <= constants.IMAGES_PER_DAY - 1:
            self.sunset_in_frames = True
        else:
            self.sunset_in_frames = False

    def __init__(self, place, times, img_paths, sunrise, sunset, train_test_valid, lat, lng, time_offset, mali_solar_noon):
        self.all_times = times
        self.all_img_paths = img_paths

        self.times, self.img_paths = Day.random_subset(times, img_paths)

        self.train_test_valid = train_test_valid
        self.place = place
        self.date = times[0].date()
        self.lat = lat
        self.lng = lng
        self.mali_solar_noon = mali_solar_noon
        self.time_offset = time_offset # Time zone offset in seconds.

        # Determine height / width
        # Perhaps unnecessary?
        #example_img = PIL.Image.open(img_paths[0]) # lazily loads - the entire image is not read into memory
        #self.width, self.height = example_img.size

        self.sunrise = sunrise
        self.sunset = sunset
        self.sunrise_idx, self.sunset_idx = self.get_sun_idx(self.times, sunrise, sunset)

        if self.sunrise_idx >= 0 and self.sunrise_idx <= constants.IMAGES_PER_DAY - 1:
            self.sunrise_in_frames = True
        else:
            self.sunrise_in_frames = False

        if self.sunset_idx >= 0 and self.sunset_idx <= constants.IMAGES_PER_DAY - 1:
            self.sunset_in_frames = True
        else:
            self.sunset_in_frames = False











