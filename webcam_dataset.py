import os, sys, calendar, glob, datetime, time, functools, numpy as np, constants, PIL, hashlib, torch
from day import Day
from torch.utils.data.dataset import Dataset

class WebcamData():
    def compare_data_types(a, b):
        return (a.train_test_valid > b.train_test_valid) - (a.train_test_valid < b.train_test_valid)

    def compare_images(a, b):
        filename1 = a.split('/')[-1]
        filename2 = b.split('/')[-1]

        end = filename1.find('_')
        filename1 = filename1[:end]

        end = filename2.find('_')
        filename2 = filename2[:end]

        date1 = datetime.datetime.strptime(filename1, "%Y-%m-%d-%H-%M-%S")
        date2 = datetime.datetime.strptime(filename2, "%Y-%m-%d-%H-%M-%S")

        return (date1 > date2) - (date1 < date2)

    def extract_times(images):
        filenames = [x.split('/')[-1] for x in images]
        dates = [x[:x.find('_')] for x in filenames]
        times = [datetime.datetime.strptime(x, "%Y-%m-%d-%H-%M-%S") for x in dates]

        return times

    def determine_train_test_valid(path):
        # Hash the directory string to determine if it falls in train/test/validation.

        path = path[path.find(path.split('/')[-7]):]  # -7 should be roundshot or panomax (data source)
        value = int(hashlib.sha256(path.encode('utf-8')).hexdigest(), 16) % constants.SPLIT_TOTAL

        if value <= constants.SPLIT_TRAIN:
            return 'train'
        elif value <= constants.SPLIT_TEST:
            return 'test'
        else:
            return 'valid'

    def load_images(self):
        if constants.CLUSTER:
            image_dir = '/srv/glusterfs/vli/data/'
        else:
            image_dir = '~/data/'
            image_dir = os.path.expanduser(image_dir)

        data = []
        for data_source in constants.DATA_SOURCES:
            curr_image_dir = image_dir + data_source + '/'

            countries = os.listdir(curr_image_dir)

            # Remove hidden folders.
            countries = [x for x in countries if not x.startswith('.')]

            for country in countries:
                country_dir = curr_image_dir + country + '/'

                places = os.listdir(country_dir)

                # Remove hidden folders.
                places = [x for x in places if not x.startswith('.')]

                for place in places:
                    place_dir = country_dir + place + '/'

                    try:
                        with open(place_dir + 'location.txt') as location_f:
                            location_split = location_f.read().split()
                            lat = float(location_split[0])
                            lng = float(location_split[1])
                    except FileNotFoundError:
                        print('WARNING - No location.txt! ' + place_dir)
                        continue

                    years = next(os.walk(place_dir))[1]

                    for year in years:
                        year_dir = place_dir + year + '/'

                        months = next(os.walk(year_dir))[1]

                        for month in months:
                            if int(month) < 4 and year == '2017': # VLI # skip anything before April 2017 for now...
                                continue

                            month_dir = year_dir + month + '/'

                            if constants.DAYS_PER_MONTH == 'MAX':
                                numDays = calendar.monthrange(int(year), int(month))[1]
                            else:
                                numDays = constants.DAYS_PER_MONTH

                            for day in range(1, numDays + 1):
                                day = "{0:0=2d}".format(day)

                                day_dir = month_dir + day + '/'

                                # May need to change if we add LARGE sizes as well. # VLI
                                train_test_valid = WebcamData.determine_train_test_valid(day_dir)

                                try:
                                    with open(day_dir + 'sun.txt') as sun_f:
                                        sun_lines = sun_f.read().splitlines()
                                except FileNotFoundError:
                                    print('WARNING - No sun.txt! ' + day_dir)
                                    continue

                                # No sunrise or no sunset this day, so skip it.
                                if sun_lines[4].find('SUN') >= 0 or sun_lines[5].find('SUN') >= 0:
                                    continue  # MAY NEED FURTHER ATTENTION # VLI

                                date_str = year + '-' + month + '-' + day
                                date = datetime.datetime.strptime(date_str, "%Y-%m-%d")

                                # Heuristic for 2 if statements below - not necessarily 100% accurate.
                                # If sunrise is from 10 PM to 11:59 PM, it happens the day before.
                                if int(sun_lines[4].split(':')[0]) >= 20:
                                    date += datetime.timedelta(days=-1)
                                    date_str = str(date.date())
                                # If sunset is from midnight to 1:59 AM, it happens the next day.
                                if int(sun_lines[5].split(':')[0]) < 2:
                                    date += datetime.timedelta(days=1)
                                    date_str = str(date.date())

                                sunrise_str = date_str + ' ' + sun_lines[4]
                                sunset_str = date_str + ' ' + sun_lines[5]
                                sunrise = datetime.datetime.strptime(sunrise_str, "%Y-%m-%d %H:%M:%S")
                                sunset = datetime.datetime.strptime(sunset_str, "%Y-%m-%d %H:%M:%S")

                                for size in constants.SIZE:
                                    image_dir = day_dir + size + '/'
                                    images = glob.glob(image_dir + '*.jpg')

                                    done = glob.glob(image_dir + '*.txt')
                                    if len(done) < 1: # no done.txt file, so skip it
                                        if year == '2018': # VLI
                                            print('WARNING - 2018 is unfinished! ' + image_dir) # VLI
                                        continue

                                    # Not enough images, so skip it.
                                    if len(images) < constants.IMAGES_PER_DAY:
                                        continue  # MAY NEED FURTHER ATTENTION # VLI

                                    # Sort by time.
                                    images.sort(key=functools.cmp_to_key(WebcamData.compare_images))

                                    # Get the list of times associated with the images.
                                    times = WebcamData.extract_times(images)

                                    # Randomly select IMAGES_PER_DAY images from times / images.
                                    subset_idx = np.random.choice(len(times), constants.IMAGES_PER_DAY,
                                                                      replace=False)
                                    subset_idx.sort()
                                    subset_times = [times[x] for x in subset_idx]
                                    subset_images = [images[x] for x in subset_idx]

                                    #test = PIL.Image.open(subset_images[5]) # VLI DELETE THIS
                                    #test.save('/home/vli/patches/original' + place + '.jpg' ) # DELETE

                                    day_obj = Day(subset_times, subset_images, sunrise, sunset,
                                                  train_test_valid)  # One training / test example.
                                    data.append(day_obj)
                                    self.types[train_test_valid] += 1

        sort_t0 = time.time()
        data.sort(key=functools.cmp_to_key(WebcamData.compare_data_types)) # Sorted in order of test, train, valid
        sort_t1 = time.time()
        print('Sorting filenames time (s): ' + str(sort_t1 - sort_t0))
        sys.stdout.flush()

        return data

    def __init__(self):
        self.types = {'train': 0, 'test': 0, 'valid':0}

        load_t0 = time.time()
        self.days = self.load_images()
        load_t1 = time.time()
        print('Load File Paths Time (s): ' + str(load_t1 - load_t0))

class Train(Dataset):
    def __init__(self, data, transforms=None):
        num_test = data.types['test']
        num_train = data.types['train']
        self.data = data.days[num_test:num_test + num_train]
        self.sunrise_label = np.asarray([x.sunrise_idx for x in self.data])
        self.sunset_label = np.asarray([x.sunset_idx for x in self.data])
        self.transforms = transforms

    def __getitem__(self, index):
        # Return image and the label
        width = self.data[index].width
        height = self.data[index].height
        img_paths = self.data[index].img_paths

        #img_stack = np.asarray([])
        img_stack = [0] * constants.IMAGES_PER_DAY
        for i, image in enumerate(img_paths):
            img = np.asarray(PIL.Image.open(image)) #cv2.imread(image)
            #cv2.imwrite('/home/vli/test.jpg', img)
            img_stack[i] = img
            #img_stack = np.stack(img_stack, img), axis=2) if img_stack.size else img # should this be 3D stack or 4D?
        #img_stack = np.stack(img_stack, axis=0)

        #print(img_stack.shape)
        #frogs = img_stack.reshape(height, width, 32 * 3)
        #print(np.shares_memory(img_stack, frogs))

        if self.transforms is not None:
            img_stack = self.transforms(img_stack)

        if constants.LEARNING_SUNRISE:
            return (img_stack, self.sunrise_label[index])
        else:
            return (img_stack, self.sunset_label[index])

    def __len__(self):
        return len(self.data)

class Test(Dataset):
    def __init__(self, data, transforms=None):
        num_test = data.types['test']
        self.data = data.days[:num_test]
        self.sunrise_label = np.asarray([x.sunrise_idx for x in self.data])
        self.sunset_label = np.asarray([x.sunset_idx for x in self.data])
        self.transforms = transforms

    def __getitem__(self, index):
        # Return image and the label
        img_paths = self.data[index].img_paths

        img_stack = [0] * constants.IMAGES_PER_DAY
        for i, image in enumerate(img_paths):
            img = np.asarray(PIL.Image.open(image)) #cv2.imread(image)
            img_stack[i] = img
        img_stack = np.stack(img_stack, axis=0)

        if self.transforms is not None:
            img_stack = self.transforms(img_stack)

        if constants.LEARNING_SUNRISE:
            return (img_stack, self.sunrise_label[index])
        else:
            return (img_stack, self.sunset_label[index])

    def __len__(self):
        return len(self.data)

class Validation(Dataset):
    def __init__(self, data, transforms=None):
        num_test = data.types['test']
        num_train = data.types['train']
        self.data = data.days[num_train + num_test:]
        self.sunrise_label = np.asarray([x.sunrise_idx for x in self.data])
        self.sunset_label = np.asarray([x.sunset_idx for x in self.data])
        self.transforms = transforms

    def __getitem__(self, index):
        # Return image and the label
        img_paths = self.data[index].img_paths

        img_stack = [0] * constants.IMAGES_PER_DAY
        for i, image in enumerate(img_paths):
            img = np.asarray(PIL.Image.open(image)) #cv2.imread(image)
            img_stack[i] = img
        img_stack = np.stack(img_stack, axis=0)

        if self.transforms is not None:
            img_stack = self.transforms(img_stack)

        if constants.LEARNING_SUNRISE:
            return (img_stack, self.sunrise_label[index])
        else:
            return (img_stack, self.sunset_label[index])

    def __len__(self):
        return len(self.data)