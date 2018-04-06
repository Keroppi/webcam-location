import os, sys, cv2, calendar, glob, datetime, time, functools, numpy as np, constants, PIL, hashlib, torch
from day import Day
from torch.utils.data.dataset import Dataset

class WebcamDataset(Dataset):
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

    def load_images():
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

                                    train_test_valid = WebcamDataset.determine_train_test_valid(day_dir)

                                    with open(day_dir + 'sun.txt') as sun_f:
                                        sun_lines = sun_f.read().splitlines()
                                        sunrise_str = sun_lines[4]
                                        sunset_str = sun_lines[5]
                                        sunrise = datetime.datetime.strptime(sunrise_str, "%Y-%m-%d %H:%M:%S")
                                        sunset = datetime.datetime.strptime(sunset_str, "%Y-%m-%d %H:%M:%S")

                                    for size in constants.SIZE:
                                        img_stack = np.array([])  # Stack all images along the color channel depth.
                                        image_dir = day_dir + size + '/'
                                        images = glob.glob(image_dir + '*.jpg')

                                        # Not enough images!
                                        if len(images) < constants.IMAGES_PER_DAY:
                                            # print("NOT ENOUGH IMAGES!")
                                            continue  # MAY NEED FURTHER ATTENTION # VLI

                                        # Sort by time.
                                        images.sort(key=functools.cmp_to_key(WebcamDataset.compare_images))

                                        # Get the list of times associated with the images.
                                        times = WebcamDataset.extract_times(images)

                                        # Randomly select IMAGES_PER_DAY images from times / images.
                                        subset_idx = np.random.choice(len(times), constants.IMAGES_PER_DAY,
                                                                      replace=False)
                                        subset_idx.sort()
                                        subset_times = [times[x] for x in subset_idx]
                                        subset_images = [images[x] for x in subset_idx]

                                        for image in subset_images:
                                            img = cv2.imread(image)
                                            # print(img.shape)
                                            # cv2.imwrite('/home/vli/test.jpg', img)

                                            img_stack = np.dstack((img_stack, img)) if img_stack.size else img

                                        day_obj = Day(subset_times, img_stack, sunrise, sunset,
                                                      train_test_valid)  # One training / test example.
                                        data.append(day_obj)

        return data

    def __init__(self, transforms=None):
        self.data = WebcamDataset.load_images()
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