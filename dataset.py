import os, sys, cv2, skimage

CLUSTER = False
DAYS_PER_MONTH = 1

if CLUSTER:
    image_dir = '/scratch_net/biwidl103/vli/data/'
else:
    image_dir = '~/data/'
    image_dir = os.path.expanduser(image_dir)

data_sources = ['roundshot'] # 'panomax'

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

                        days = next(os.walk(month_dir))[1]

                        print(days)



