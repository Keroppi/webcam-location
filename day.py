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


    def in_frames(self, times, sunrise, sunset):
        all_sunrise_idx, all_sunset_idx = self.get_sun_idx(times, sunrise, sunset)

        if all_sunrise_idx >= 0 and all_sunrise_idx < len(times):
            sunrise_in_frames = True
        else:
            sunrise_in_frames = False

        if all_sunset_idx >= 0 and all_sunset_idx < len(times):
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

    def change_frames(self, center_frame, mode='sunrise'): # Given a suggested frame idx, repick frames that are close to it.
        suggested_time = self.get_local_time(center_frame)

        start_idx = len(self.all_times) - 1
        end_idx = len(self.all_times) - 1
        for t_idx, time in enumerate(self.all_times):
            if suggested_time < time:
                start_idx = max(t_idx - 1, 0)
                end_idx = t_idx
                break

        #if center_frame < 0:
        #    start = 0
        #    end = 0
        #elif center_frame >= constants.IMAGES_PER_DAY - 1:
        #    start = constants.IMAGES_PER_DAY - 1
        #    end = constants.IMAGES_PER_DAY - 1
        #else:
        #    start = math.floor(center_frame)
        #    end = math.ceil(center_frame)

        #start_pivot_time = self.times[start]
        #end_pivot_time = self.times[end]

        #for t_idx, time in enumerate(self.all_times):
        #    if time == start_pivot_time:
        #        start_idx = t_idx # max(t_idx, 0)
        #    if time == end_pivot_time:
        #        end_idx = t_idx # min(t_idx, len(self.all_times) - 1) # Inclusive

        center_idx = round((start_idx + end_idx) / 2)
        start_idx = center_idx - math.floor((constants.IMAGES_PER_DAY - 1) / 2)
        end_idx = center_idx + math.ceil((constants.IMAGES_PER_DAY - 1) / 2)

        # VLI
        if mode == 'sunrise':
            diff = abs((self.sunrise - self.all_times[center_idx]).total_seconds() / 3600)
            if diff < 0.5:
                print('SUNRISE GUESS LESS THAN 0.5 HOUR OFF')
                print(self.sunrise)
                print(self.all_times[center_idx])
        else:
            diff = abs((self.sunset - self.all_times[center_idx]).total_seconds() / 3600)
            if diff < 0.5:
                print('SUNSET GUESS LESS THAN 0.5 HOUR OFF')
                print(self.sunset)
                print(self.all_times[center_idx])
        # END VLI

        if start_idx < 0:
            start_idx = 0
            end_idx = constants.IMAGES_PER_DAY - 1
        if end_idx > len(self.all_times) - 1:
            start_idx = len(self.all_times) - 1 - (constants.IMAGES_PER_DAY - 1)
            end_idx = len(self.all_times) - 1

        #num_important_frames = end_idx - start_idx + 1
        #if num_important_frames > constants.IMAGES_PER_DAY:
        #    end_idx = start_idx + constants.IMAGES_PER_DAY - 1

        #important_frames = list(range(start_idx, end_idx + 1))
        #remaining = set(range(len(self.all_times))) - set(important_frames)

        #subset_idx = np.random.choice(len(remaining), constants.IMAGES_PER_DAY - len(important_frames), replace=False)
        #subset_idx.sort()
        #subset_idx = [list(remaining)[x] for x in subset_idx]
        #subset_idx = subset_idx + important_frames
        #subset_idx.sort()

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
                if not (self.sunrise_idx >= 0 and self.sunrise_idx < constants.IMAGES_PER_DAY):
                    sunrise_before = False
                else:
                    sunrise_before = True
        else:
            if self.sunset_in_frames:
                if not (self.sunset_idx >= 0 and self.sunset_idx < constants.IMAGES_PER_DAY):
                    sunset_before = False

                    print('SUNSET WAS BAD BEFORE')  # Can count number of times this phrase appears.
                    print(self.sunset)
                    print(self.times[0])
                    print(self.times[-1])
                    print(self.all_times[0])
                    print(self.all_times[-1])
                    print(self.all_times[center_idx])
                else:
                    sunset_before = True

                    print('SUNSET WAS GOOD BEFORE')  # Can count number of times this phrase appears.
                    print(self.sunset)
                    print(self.times[0])
                    print(self.times[-1])
                    print(self.all_times[0])
                    print(self.all_times[-1])
                    print(self.all_times[center_idx])

        #### END VLI

        self.times = [self.all_times[x] for x in subset_idx]
        self.img_paths = [self.all_img_paths[x] for x in subset_idx]

        self.sunrise_idx, self.sunset_idx = self.get_sun_idx(self.times, self.sunrise, self.sunset)

        if mode == 'sunrise': # VLI remove mode as well from args - this is purely for debugging
            if self.sunrise_in_frames:
                if not (self.sunrise_idx >= 0 and self.sunrise_idx < constants.IMAGES_PER_DAY):
                    if sunrise_before:
                        print('CHANGE FRAMES MADE SUNRISE WORSE') # Can count number of times this phrase appears.
                else:
                    if not sunrise_before:
                        print('CHANGE FRAMES MADE SUNRISE BETTER') # Can count number of times this phrase appears.

        else:
            if self.sunset_in_frames:
                if not (self.sunset_idx >= 0 and self.sunset_idx < constants.IMAGES_PER_DAY):
                    if sunset_before:
                        print('CHANGE FRAMES MADE SUNSET WORSE') # Can count number of times this phrase appears.
                        print(self.sunset)
                        print(self.times[0])
                        print(self.times[-1])
                        print(self.all_times[0])
                        print(self.all_times[-1])
                        print(self.all_times[center_idx])
                else:
                    if not sunset_before:
                        print('CHANGE FRAMES MADE SUNSET BETTER') # Can count number of times this phrase appears.
        sys.stdout.flush()
        ### END VLI


    def random_frames(self):
        self.times, self.img_paths = self.random_subset(self.all_times, self.all_img_paths)
        self.sunrise_idx, self.sunset_idx = self.get_sun_idx(self.times, self.sunrise, self.sunset)

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

        if self.sunset_in_frames and train_test_valid == 'test': # VLI
            print('SUNSET IN FRAMES')
            sys.stdout.flush()

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












