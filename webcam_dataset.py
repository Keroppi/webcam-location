import os, sys, cv2, calendar, glob, datetime, time, functools, numpy as np, constants, PIL, hashlib, torch
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
            image_dir = '/scratch_net/biwidl103/vli/data/'
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

                    with open(place_dir + 'location.txt') as location_f:
                        location_split = location_f.read().split()
                        lat = float(location_split[0])
                        lng = float(location_split[1])

                        years = next(os.walk(place_dir))[1]

                        for year in years:
                            year_dir = place_dir + year + '/'

                            months = next(os.walk(year_dir))[1]

                            for month in months:
                                month_dir = year_dir + month + '/'

                                if constants.DAYS_PER_MONTH == 'MAX':
                                    numDays = calendar.monthrange(int(year), int(month))[1]
                                else:
                                    numDays = constants.DAYS_PER_MONTH

                                for day in range(1, numDays + 1):
                                    day = "{0:0=2d}".format(day)

                                    day_dir = month_dir + day + '/'

                                    train_test_valid = WebcamData.determine_train_test_valid(day_dir)
                                    self.types[train_test_valid] += 1

                                    with open(day_dir + 'sun.txt') as sun_f:
                                        sun_lines = sun_f.read().splitlines()
                                        sunrise_str = sun_lines[4]
                                        sunset_str = sun_lines[5]
                                        sunrise = datetime.datetime.strptime(sunrise_str, "%Y-%m-%d %H:%M:%S")
                                        sunset = datetime.datetime.strptime(sunset_str, "%Y-%m-%d %H:%M:%S")

                                    for size in constants.SIZE:
                                        #img_stack = np.array([])  # Stack all images along the color channel depth.
                                        image_dir = day_dir + size + '/'
                                        images = glob.glob(image_dir + '*.jpg')

                                        # Not enough images!
                                        if len(images) < constants.IMAGES_PER_DAY:
                                            # print("NOT ENOUGH IMAGES!")
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

                                        day_obj = Day(subset_times, subset_images, sunrise, sunset,
                                                      train_test_valid)  # One training / test example.
                                        data.append(day_obj)

        data.sort(key=functools.cmp_to_key(WebcamData.compare_data_types)) # Sorted in order of test, train, valid
        return data

    def __init__(self):
        self.types = {'train': 0, 'test': 0, 'valid':0}
        self.days = self.load_images()

class Train(Dataset):
    def __init__(self, data, transforms=None):
        num_test = data.types['test']
        num_train = data.types['train']
        self.data = data.days[num_test:num_train]
        self.sunrise_label = np.asarray([x.sunrise_idx for x in self.data])
        self.sunset_label = np.asarray([x.sunset_idx for x in self.data])
        self.transforms = transforms

    def __getitem__(self, index):
        # Return image and the label
        #width = self.data[index].width
        #height = self.data[index].height
        img_paths = self.data[index].img_paths

        #img_stack = np.asarray([])
        img_stack = [0] * constants.IMAGES_PER_DAY
        for i, image in enumerate(img_paths):
            img = cv2.imread(image)
            #cv2.imwrite('/home/vli/test.jpg', img)
            img_stack[i] = img
            #img_stack = np.stack(img_stack, img), axis=2) if img_stack.size else img # should this be 3D stack or 4D?
        img_stack = np.stack(img_stack, axis=-1)

        if self.transforms is not None:
            img_stack = self.transforms(img_stack)

        return (img_stack, self.sunrise_label[index]) # SKIP SUNSET FOR NOW # VLI

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

        data = self.data[index].img_stack
        if self.transforms is not None:
            data = self.transforms(data)

        return (data, self.sunrise_label[index]) # SKIP SUNSET FOR NOW # VLI

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

        data = self.data[index].img_stack
        if self.transforms is not None:
            data = self.transforms(data)

        return (data, self.sunrise_label[index]) # SKIP SUNSET FOR NOW # VLI

    def __len__(self):
        return len(self.data)