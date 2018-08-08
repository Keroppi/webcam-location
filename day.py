import constants
import numpy as np, sys, PIL, math, time, datetime, random, statistics

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
            return self.times[0] + idx * diff / (constants.IMAGES_PER_DAY - 1) # UNDO SCALING?
        else:
            diff = self.times[-1] - self.times[0]
            return self.times[constants.IMAGES_PER_DAY - 1] + (idx - (constants.IMAGES_PER_DAY - 1)) * diff / (constants.IMAGES_PER_DAY - 1) # UNDO SCALING?

    def get_sun_idx(self, times, sunrise, sunset):
        sunrise_idx = 0
        sunset_idx = 0

        # Time between first and last images.
        # Used for scaling if the sunrise / sunset fall outside the range of images.
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
            extra = (sunrise - times[-1]) / time_diff * (constants.IMAGES_PER_DAY - 1) # UNDO SCALING?
            sunrise_idx += extra
        elif max_sunrise_idx == 0: # Before the first image.
            sunrise_idx = 0
            extra = (sunrise - times[0]) / time_diff * (constants.IMAGES_PER_DAY - 1) # UNDO SCALING?
            sunrise_idx += extra
        else:
            remainder = (sunrise - times[max_sunrise_idx - 1]) / (times[max_sunrise_idx] - times[max_sunrise_idx - 1])
            sunrise_idx = remainder + max_sunrise_idx - 1

        if max_sunset_idx is None:
            sunset_idx = constants.IMAGES_PER_DAY - 1
            extra = (sunset - times[-1]) / time_diff * (constants.IMAGES_PER_DAY - 1) # UNDO SCALING?
            sunset_idx += extra
        elif max_sunset_idx == 0:
            sunset_idx = 0
            extra = (sunset - times[0]) / time_diff * (constants.IMAGES_PER_DAY - 1) # UNDO SCALING?
            sunset_idx += extra
        else:
            remainder = (sunset - times[max_sunset_idx - 1]) / (times[max_sunset_idx] - times[max_sunset_idx - 1])
            sunset_idx = remainder + max_sunset_idx - 1

        #'''
        print('Sunrise / Sunset and Idx')
        print(sunrise_idx)
        print(sunrise)
        if sunrise_idx >= 0:
            print(times[math.floor(sunrise_idx)])
            print(times[math.ceil(sunrise_idx)])
        print('')

        print(sunset_idx)
        print(sunset)
        if sunset_idx <= len(times) - 1:
            print(times[math.floor(sunset_idx)])
            print(times[math.ceil(sunset_idx)])
        print('')
        print(time_diff)
        print(times[0])
        print(times[-1])
        print('')
        #'''

        return (sunrise_idx, sunset_idx)


    def in_frames(self, times, sunrise, sunset):
        all_sunrise_idx, all_sunset_idx = self.get_sun_idx(times, sunrise, sunset)

        if all_sunrise_idx >= 0 and all_sunrise_idx <= len(times) - 1:
            sunrise_in_frames = True
        else:
            sunrise_in_frames = False

        if all_sunset_idx >= 0 and all_sunset_idx <= len(times) - 1:
            sunset_in_frames = True
        else:
            sunset_in_frames = False

        return (sunrise_in_frames, sunset_in_frames)

    def random_subset(self, all_times, all_img_paths):
        # Randomly select IMAGES_PER_DAY images from times / images.
        subset_idx = np.random.choice(len(all_times), constants.IMAGES_PER_DAY, replace=False)
        subset_idx.sort()
        times = [all_times[x] for x in subset_idx]
        img_paths = [all_img_paths[x] for x in subset_idx]

        return (times, img_paths)

    def uniform_subset(self, all_times, all_img_paths):
        # Uniformly select IMAGES_PER_DAY images from times / images.
        f = lambda m, n: [i * n // m + n // (2 * m) for i in range(m)]
        subset_idx = f(constants.IMAGES_PER_DAY, len(all_times))
        subset_idx[0] = 0
        subset_idx[-1] = len(all_times) - 1
        #subset_idx = np.random.choice(len(all_times), constants.IMAGES_PER_DAY, replace=False)
        #subset_idx.sort()
        times = [all_times[x] for x in subset_idx]
        img_paths = [all_img_paths[x] for x in subset_idx]

        return (times, img_paths)

    def change_frames_medium(self, center_frame, mode='sunrise', scale_factor=1.5, pass_idx=1, reverse=False): # Given a suggested frame idx, uniformly pick from 97 frames that are closest to it.
        suggested_time = self.get_local_time(center_frame)

        start_idx = len(self.all_times) - 1
        end_idx = len(self.all_times) - 1
        for t_idx, time in enumerate(self.all_times):
            if suggested_time < time:
                start_idx = max(t_idx - 1, 0)
                end_idx = t_idx
                break

        center_idx = round((start_idx + end_idx) / 2)
        start_idx = center_idx - math.ceil(constants.IMAGES_PER_DAY * scale_factor)
        end_idx = center_idx + math.ceil(constants.IMAGES_PER_DAY * scale_factor) # 97 frames if scale_factor = 1.5, # 289 frames if scale_factor = 4.5

        if start_idx < 0:
            start_idx = 0
            end_idx = min(math.ceil(2 * scale_factor * constants.IMAGES_PER_DAY), len(self.all_times) - 1)
        if end_idx > len(self.all_times) - 1:
            start_idx = max(len(self.all_times) - 1 - math.ceil(2 * scale_factor * constants.IMAGES_PER_DAY), 0)
            end_idx = len(self.all_times) - 1

        if mode == 'sunrise':
            if self.sunrise_in_frames:
                if not (self.sunrise_idx >= 0 and self.sunrise_idx <= constants.IMAGES_PER_DAY - 1):
                    sunrise_before = False
                else:
                    sunrise_before = True
        else:
            if self.sunset_in_frames:
                if not (self.sunset_idx >= 0 and self.sunset_idx <= constants.IMAGES_PER_DAY - 1):
                    sunset_before = False

                    print('SUNSET WAS BAD BEFORE')  # Can count number of times this phrase appears.
                    print(self.sunset)
                    print(self.times[0])
                    print(self.times[-1])
                    print(self.all_times[0])
                    print(self.all_times[-1])
                    print(self.all_times[center_idx])
                    print('')
                    sys.stdout.flush()
                else:
                    sunset_before = True


        self.times, self.img_paths = self.uniform_subset(self.all_times[start_idx:end_idx + 1],
                                                         self.all_img_paths[start_idx:end_idx + 1])

        if reverse:
            self.reverse_images()

        self.sunrise_idx, self.sunset_idx = self.get_sun_idx(self.times, self.sunrise, self.sunset)

        if mode == 'sunrise':
            if self.sunrise_in_frames:
                if not (self.sunrise_idx >= 0 and self.sunrise_idx <= constants.IMAGES_PER_DAY - 1):
                    if sunrise_before:
                        print('CHANGE FRAMES {} MADE SUNRISE WORSE'.format(pass_idx)) # Can count number of times this phrase appears.
                    else:
                        print('CHANGE FRAMES {} DID NOT IMPROVE SUNRISE'.format(pass_idx))
                else:
                    if not sunrise_before:
                        print('CHANGE FRAMES {} MADE SUNRISE BETTER'.format(pass_idx))
        else:
            if self.sunset_in_frames:
                if not (self.sunset_idx >= 0 and self.sunset_idx <= constants.IMAGES_PER_DAY - 1):
                    if sunset_before:
                        print('CHANGE FRAMES {} MADE SUNSET WORSE'.format(pass_idx))
                    else:
                        print('CHANGE FRAMES {} DID NOT IMPROVE SUNSET'.format(pass_idx))
                        print(self.sunset)
                        print(self.times[0])
                        print(self.times[-1])
                        print(self.all_times[0])
                        print(self.all_times[-1])
                        print(self.all_times[center_idx])
                        print('')
                        sys.stdout.flush()
                else:
                    if not sunset_before:
                        print('CHANGE FRAMES {} MADE SUNSET BETTER'.format(pass_idx))


        sys.stdout.flush()

    def change_frames_fine(self, center_frame, mode='sunrise', reverse=False): # Given a suggested frame idx, repick 32 frames that are closest to it.
        suggested_time = self.get_local_time(center_frame)

        start_idx = len(self.all_times) - 1
        end_idx = len(self.all_times) - 1
        for t_idx, time in enumerate(self.all_times):
            if suggested_time < time:
                start_idx = max(t_idx - 1, 0)
                end_idx = t_idx
                break

        center_idx = round((start_idx + end_idx) / 2)
        start_idx = center_idx - math.floor((constants.IMAGES_PER_DAY - 1) / 2)
        end_idx = center_idx + math.ceil((constants.IMAGES_PER_DAY - 1) / 2)

        '''
        if mode == 'sunrise':
            diff = abs((self.sunrise - self.all_times[center_idx]).total_seconds() / 3600)
            if diff < 0.5:
                print('SUNRISE GUESS LESS THAN 0.5 HOUR OFF')
                #print(self.sunrise)
                #print(self.all_times[center_idx])
        else:
            diff = abs((self.sunset - self.all_times[center_idx]).total_seconds() / 3600)
            if diff < 0.5:
                print('SUNSET GUESS LESS THAN 0.5 HOUR OFF')
                #print(self.sunset)
                #print(self.all_times[center_idx])
        '''

        if start_idx < 0:
            start_idx = 0
            end_idx = constants.IMAGES_PER_DAY - 1
        if end_idx > len(self.all_times) - 1:
            start_idx = len(self.all_times) - 1 - (constants.IMAGES_PER_DAY - 1)
            end_idx = len(self.all_times) - 1

        subset_idx = list(range(start_idx, end_idx + 1))

        '''
        print(center_frame)
        print(self.times)
        print(self.all_times)
        print(subset_idx)
        print(len(subset_idx))
        print('')
        sys.stdout.flush()
        '''


        if mode == 'sunrise': # VLI remove mode as well from args - this is purely for debugging
            if self.sunrise_in_frames:
                if not (self.sunrise_idx >= 0 and self.sunrise_idx <= constants.IMAGES_PER_DAY - 1):
                    sunrise_before = False
                else:
                    sunrise_before = True
        else:
            if self.sunset_in_frames:
                if not (self.sunset_idx >= 0 and self.sunset_idx <= constants.IMAGES_PER_DAY - 1):
                    sunset_before = False
                    '''
                    print('SUNSET WAS BAD BEFORE')  # Can count number of times this phrase appears.
                    print(self.sunset)
                    print(self.times[0])
                    print(self.times[-1])
                    print(self.all_times[0])
                    print(self.all_times[-1])
                    print(self.all_times[center_idx])
                    '''
                else:
                    sunset_before = True
                    '''
                    print('SUNSET WAS GOOD BEFORE')  # Can count number of times this phrase appears.
                    print(self.sunset)
                    print(self.times[0])
                    print(self.times[-1])
                    print(self.all_times[0])
                    print(self.all_times[-1])
                    print(self.all_times[center_idx])
                    '''
        #### END VLI

        self.times = [self.all_times[x] for x in subset_idx]
        self.img_paths = [self.all_img_paths[x] for x in subset_idx]

        if reverse:
            self.reverse_images()

        self.sunrise_idx, self.sunset_idx = self.get_sun_idx(self.times, self.sunrise, self.sunset)

        if mode == 'sunrise': # VLI remove mode as well from args - this is purely for debugging
            if self.sunrise_in_frames:
                if not (self.sunrise_idx >= 0 and self.sunrise_idx <= constants.IMAGES_PER_DAY - 1):
                    if sunrise_before:
                        print('CHANGE FRAMES 2 MADE SUNRISE WORSE') # Can count number of times this phrase appears.
                    else:
                        print('CHANGE FRAMES 2 DID NOT IMPROVE SUNRISE')
                else:
                    if not sunrise_before:
                        print('CHANGE FRAMES 2 MADE SUNRISE BETTER') # Can count number of times this phrase appears.
        else:
            if self.sunset_in_frames:
                if not (self.sunset_idx >= 0 and self.sunset_idx <= constants.IMAGES_PER_DAY - 1):
                    if sunset_before:
                        print('CHANGE FRAMES 2 MADE SUNSET WORSE') # Can count number of times this phrase appears.
                    else:
                        print('CHANGE FRAMES 2 DID NOT IMPROVE SUNSET')
                        #print(self.sunset)
                        #print(self.times[0])
                        #print(self.times[-1])
                        #print(self.all_times[0])
                        #print(self.all_times[-1])
                        #print(self.all_times[center_idx])
                        #print('')
                else:
                    if not sunset_before:
                        print('CHANGE FRAMES 2 MADE SUNSET BETTER') # Can count number of times this phrase appears.

        sys.stdout.flush()
        ### END VLI


    def random_frames(self):
        self.times, self.img_paths = self.random_subset(self.all_times, self.all_img_paths)
        self.sunrise_idx, self.sunset_idx = self.get_sun_idx(self.times, self.sunrise, self.sunset)

    def reverse_images(self):
        self.img_paths = list(reversed(self.img_paths))

    def uniform_frames(self, reverse=False):
        self.times, self.img_paths = self.uniform_subset(self.all_times, self.all_img_paths)

        if reverse:
            self.reverse_images()

        self.sunrise_idx, self.sunset_idx = self.get_sun_idx(self.times, self.sunrise, self.sunset)

    def reverse_get_local_time(self, idx):
        idx = constants.IMAGES_PER_DAY - 1 - idx
        return self.get_local_time(idx)

        #19:30 sunset
        #17:00, 18:00, 20:00 - 1.75 idx normally
        # 20, 18, 17 - reverse outputs 0.25 -> 3 - 1 - 0.25 = 1.75

        # 14:00 sunset
        # 17:00, 18:00, 20:00 -> -0.33333333 idx normally
        # 20, 18, 17 - reverse outputs 2.3333333 -> 3 - 1 - 2.333333 = -0.3333333

    def reverse_change_frames_medium(self, center_frame, mode='sunrise', scale_factor=1.5, pass_idx=1):
        center_frame = constants.IMAGES_PER_DAY - 1 - center_frame
        self.change_frames_medium(center_frame, mode, scale_factor, pass_idx, True)

    def reverse_change_frames_fine(self, center_frame, mode='sunrise'):
        center_frame = constants.IMAGES_PER_DAY - 1 - center_frame
        self.change_frames_fine(center_frame, mode, True)

    def __init__(self, place, times, img_paths, sunrise, sunset, train_test_valid, lat, lng, time_offset, mali_solar_noon):
        self.all_times = times
        self.all_img_paths = img_paths

        self.times, self.img_paths = self.random_subset(times, img_paths)

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

        self.sunrise_in_frames, self.sunset_in_frames = self.in_frames(self.all_times, self.sunrise, self.sunset)

        #if self.sunset_in_frames and train_test_valid == 'test':
        #    print('SUNSET IN FRAMES')
        #    sys.stdout.flush()

        diff = [self.all_times[idx] - self.all_times[idx - 1] for idx, _ in enumerate(self.all_times) if idx > 0]
        diff_min = [x.total_seconds() / 60 for x in diff]
        self.interval_min = statistics.mean(diff_min)

        if self.lat > 0: # Northern hemisphere
            if self.date.month >= 3 and self.date.month <= 5:
                self.season = 'spring'
            elif self.date.month >= 6 and self.date.month <= 8:
                self.season = 'summer'
            elif self.date.month >= 9 and self.date.month <= 11:
                self.season = 'fall'
            else:
                self.season = 'winter'
        else: # Southern hemisphere
            if self.date.month >= 3 and self.date.month <= 5:
                self.season = 'fall'
            elif self.date.month >= 6 and self.date.month <= 8:
                self.season = 'winter'
            elif self.date.month >= 9 and self.date.month <= 11:
                self.season = 'spring'
            else:
                self.season = 'summer'