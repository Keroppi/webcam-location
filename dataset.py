import os, sys, cv2, calendar, glob, datetime, time, functools, numpy as np, constants
from day import Day

if constants.CLUSTER:
    image_dir = '/scratch_net/biwidl103/vli/data/'
else:
    image_dir = '~/data/'
    image_dir = os.path.expanduser(image_dir)

data_sources = ['roundshot'] # 'panomax'

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


for data_source in data_sources:
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

                        img_stack = np.array([]) # Stack all images along the color channel depth.
                        for day in range(1, numDays + 1):
                            day = "{0:0=2d}".format(day)

                            day_dir = month_dir + day + '/'

                            with open(day_dir + 'sun.txt') as sun_f:
                                sun_lines = sun_f.read().splitlines()
                                sunrise_str = sun_lines[4]
                                sunset_str = sun_lines[5]
                                sunrise = datetime.datetime.strptime(sunrise_str, "%Y-%m-%d %H:%M:%S")
                                sunset = datetime.datetime.strptime(sunset_str, "%Y-%m-%d %H:%M:%S")

                            for size in constants.SIZE:
                                image_dir = day_dir + size + '/'
                                images = glob.glob(image_dir + '*.jpg')

                                # Not enough images!
                                if len(images) < constants.IMAGES_PER_DAY:
                                    #print("NOT ENOUGH IMAGES!")
                                    continue # MAY NEED FURTHER ATTENTION # VLI

                                # Sort by time.
                                images.sort(key=functools.cmp_to_key(compare_images))

                                # Get the list of times associated with the images.
                                times = extract_times(images)

                                for image in images:
                                    img = cv2.imread(image)
                                    #cv2.imwrite('/home/vli/test.jpg', img)

                                    img_stack = np.dstack((img_stack, img)) if img_stack.size else img

                            day_obj = Day(times, img_stack, sunrise, sunset) # One training / test example.







