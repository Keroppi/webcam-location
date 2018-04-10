#!/scratch_net/biwidl106/vli/.pyenv/shims/python

import string, sys, os, requests, urllib.request, json, multiprocessing as mp, datetime, time, traceback, socket, random, glob
from urllib.error import ContentTooShortError
from http.client import RemoteDisconnected
from urllib.request import URLError
from urllib.request import HTTPError
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

CLUSTER = True # run on cluster or local machine
SIZE = 'small' # 'large'

GOOGLE_MAPS_API_KEY = 'AIzaSyCEJkK4hEYYnRv4z6hL6n8A8VqfqJdspnY';

if CLUSTER:
    SGE_TASK_ID = int(os.environ.get('SGE_TASK_ID')) # Determines the month that gets downloaded. (1 -> January)
    baseLocation = '/scratch_net/biwidl103/vli/data/roundshot/'
else:
    LOCAL_MONTH = 1 # January
    baseLocation = '~/data/roundshot/'
    baseLocation = os.path.expanduser(baseLocation)

year = '2018'

rs_filename = '~/roundshot.txt'
rs_filename = os.path.expanduser(rs_filename)
    
baseUrl = 'http://storage.roundshot.com/';

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:55.0) Gecko/20100101 Firefox/55.0')]
urllib.request.install_opener(opener)

with open(rs_filename) as rs_file:
    lines = rs_file.read().splitlines()


def download_day(country, name, lat, lng, storage_id, year, month, day):
 try:
    day = "{0:0=2d}".format(day)
    
    #print('Day: ' + day)
    #sys.stdout.flush()

    date = year + '-' + month + '-' + day
    url = baseUrl + storage_id + '/'
    wrtPth =  baseLocation + country + '/' + name + '/' + year + '/' + month + '/' + day
    wrtPthSmall = wrtPth + '/small'
    wrtPthLarge = wrtPth + '/large'
    
    try:
        os.makedirs(wrtPthSmall);                
        os.makedirs(wrtPthLarge);
    except FileExistsError:
        pass

    '''
    # If file already exists and has size > 0, skip this.
    sun_file_str = wrtPth + '/sun.txt'
    if not (os.path.isfile(sun_file_str) and os.path.getsize(sun_file_str) > 0):    
        # Get the time zone offset (for local time).
        d = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        timestamp = time.mktime(d.timetuple()) + 12 * 60
        google_url = 'https://maps.googleapis.com/maps/api/timezone/json?location=' + str(lat) + ',' + str(lng) + '&timestamp=' + str(timestamp) + '&key=' + GOOGLE_MAPS_API_KEY

        retries = 0
        while True:
            with urllib.request.urlopen(google_url) as goog_url_obj:
                time_data = json.loads(goog_url_obj.read().decode())

                if time_data['status'] == 'OK':
                    break
                else:
                    if retries < 600:
                        time.sleep(2) # Possibly rate limited.
                        retries += 1
                    else:
                        print('Exceed Google API quota - sleeping.') # Exceed quota for the day.
                        sys.stdout.flush()
                        time.sleep(3600) # 1 hour
                        retries = 0

        offset = time_data['dstOffset'] + time_data['rawOffset']; # seconds
    
        # Get sunrise, sunset, and day length (API in UTC, not
        # local time - have to add it back in).
        sun_url = 'http://api.sunrise-sunset.org/json?lat=' + str(lat) + '&lng=' + str(lng) + '&date=' + date

        while True:
            try:
                with urllib.request.urlopen(sun_url) as sun_url_obj:
                    sun_json_response = sun_url_obj.read()                        
                    sun_data = json.loads(sun_json_response.decode())

                    if sun_data['status'] == 'OK':
                        break
                    else:
                        time.sleep(2)
            except HTTPError:
                time.sleep(2)
            
        utc_sunrise = date + ' ' + sun_data['results']['sunrise']
        utc_sunrise = datetime.datetime.strptime(utc_sunrise, "%Y-%m-%d %I:%M:%S %p")        
        local_sunrise = utc_sunrise + datetime.timedelta(seconds=offset)
        local_sunrise_str = str(local_sunrise)
        local_sunrise = datetime.datetime.strptime(local_sunrise_str, "%Y-%m-%d %H:%M:%S")
        local_sunrise_str = str(local_sunrise)
    
        utc_sunset = date + ' ' + sun_data['results']['sunset']
        utc_sunset = datetime.datetime.strptime(utc_sunset, "%Y-%m-%d %I:%M:%S %p")        
        local_sunset = utc_sunset + datetime.timedelta(seconds=offset)
        local_sunset_str = str(local_sunset)
        local_sunset = datetime.datetime.strptime(local_sunset_str, "%Y-%m-%d %H:%M:%S")
        local_sunset_str = str(local_sunset)

        day_length_split = [int(x) for x in sun_data['results']['day_length'].split(':')];
        day_length_seconds = str(day_length_split[0] * 3600 + day_length_split[1] * 60 + day_length_split[2]);
    
        # Save UTC (first 2 rows), day length (HH:MM:SS), 
        # timezone offset (sec), local time, and day length (sec).
        with open(wrtPth + '/sun.txt', 'w') as sun_file:
            sun_file.write(sun_data['results']['sunrise'] + '\n')
            sun_file.write(sun_data['results']['sunset'] + '\n')
            sun_file.write(sun_data['results']['day_length'] + '\n')
            sun_file.write(str(offset) + '\n')
            sun_file.write(local_sunrise_str + '\n')
            sun_file.write(local_sunset_str + '\n')
            sun_file.write(day_length_seconds + '\n')
    '''

    # Find where we left off.    
    small_files = glob.glob(wrtPthSmall + '/*')

    if wrtPthSmall + '/done.txt' in small_files:
        return

    if len(small_files) > 0:
        small_latest_file = max(small_files, key=os.path.getctime)
        hourStart = small_latest_file.split('/')[-1].split('-')[3]
        minuteStart = small_latest_file.split('/')[-1].split('-')[4]
        hourStart = int(hourStart)
        minuteStart = int(minuteStart)
    else:
        hourStart = 0
        minuteStart = 0

    # TO DO: LARGE FILES    

    for idxHr in range(hourStart, 24):
        for idxMnt in range(minuteStart, 60): 
            hour = "{0:0=2d}".format(idxHr)
            minute = "{0:0=2d}".format(idxMnt)            
            img_time = hour + '-' + minute + '-00'
            
            imgUrlSmall = url + date + '/' + img_time + '/' + date + '-' + img_time + '_thumbnail.jpg'
            imgUrlLarge = url + date + '/' + img_time + '/' + date + '-' + img_time + '_full.jpg'
            localPathSmall = wrtPthSmall + '/' + date + '-' + img_time + '_thumbnail.jpg'
            localPathLarge = wrtPthLarge + '/' + date + '-' + img_time + '_full.jpg'

            urlE = None
            keep_going = True
            if SIZE == 'small':
                #small_img_t0 = time.time()
                while keep_going:
                    try:
                        urllib.request.urlretrieve(imgUrlSmall, localPathSmall)
                        #img = urllib.request.urlopen(imgUrlSmall, timeout=0.00001).read()
                        #with open(localPathSmall, 'wb') as small_file:
                        #    small_file.write(img)

                        keep_going = False
                    except HTTPError as e: # 404 (sometimes 503)
                        #print(imgUrlSmall + ' failed with code: ' + str(e.code))
                        keep_going = False
                    except URLError as e: # timeout, but file exists...
                        if str(e.reason) != 'timed out': # image doesn't exist
                            keep_going = False
                            pass
                        else: # try again due to timeout - very unlikely
                            pass
                    except RemoteDisconnected as e: # try again?
                        print('Remote Disconnected')
                        print(imgUrlSmall)
                        sys.stdout.flush()
                    except ContentTooShortError as e: # partial download, try again
                        print('Content Too Short Error')
                        print(imgUrlSmall)
                        sys.stdout.flush()
                        try:
                            os.remove(localPathSmall)
                        except FileNotFoundError:
                            pass
                        
                #small_img_t1 = time.time()

                #blah = random.randint(0, 100)
                #if blah % 30 == 0:
                #    print('Image Time: ' + str(small_img_t1 - small_img_t0))
                ##sys.stdout.flush()
        
            else: # large           
                # To do
                pass

    with open(wrtPthSmall + '/done.txt', 'w') as small_path:
        small_path.write('done')

    # TO DO: LARGE FILES    

 except Exception as e:
     print('EXCEPTION')
     print(e.__class__.__name__)
     print(str(e))
     exc_type, exc_obj, exc_tb = sys.exc_info()
     traceback.print_tb(exc_tb)
     sys.stdout.flush()
            
lIdx = 0
while lIdx < len(lines):
    country = lines[lIdx];

    print('')
    print(country)
    print('')
    sys.stdout.flush()
    
    num_webcams = lines[lIdx + 1];
    num_webcams = num_webcams.split(': ');
    num_webcams = int(num_webcams[1]);

    lIdx += 2;
    
    line = lines[lIdx];
    while line != '-----':
        name = line;

        print(name)
        sys.stdout.flush()
        
        website = lines[lIdx + 1]
        lat_long = lines[lIdx + 2]
        lat_long_split = lat_long.split(' ');
        lat = float(lat_long_split[0])
        lng = float(lat_long_split[1])
        storage_id = lines[lIdx + 3]
        blank_line = lines[lIdx + 4]
        line = lines[lIdx + 5]
        
        lIdx += 5

        # Save latitude and longitude to a file.
        locWrtPth = baseLocation + country + '/' + name + '/'
        try:
            os.makedirs(locWrtPth)
        except FileExistsError:
            pass

        with open(locWrtPth + 'location.txt', 'w+') as loc_file:
            loc_file.write(lat_long + '\n')

            
        if CLUSTER:
            mtIdx = SGE_TASK_ID
        else:
            mtIdx = LOCAL_MONTH

        month_t0 = time.time()
        month = "{0:0=2d}".format(mtIdx) # 2-digits (8 --> 08)

        #print('Month: ' + month)
        #sys.stdout.flush()

        if mtIdx <= 7 and mtIdx % 2 == 1:
            numDays = 31;
        elif mtIdx <= 7:
            numDays = 30;
        elif mtIdx % 2 == 1:
            numDays = 30;
        else:
            numDays = 31;
            
        # February is a special case.
        if mtIdx == 2:
            numDays = 28;
            
            # Leap Year
            if int(year) % 4 == 0:
                numDays = 29;

        with ThreadPoolExecutor(numDays) as executor:               
            future_download = {executor.submit(download_day, country, name, lat, lng, storage_id, year, month, day): day for day in range(1, numDays + 1)}
                
            for _ in as_completed(future_download):
                pass
        month_t1 = time.time()
        print('Month Time (min): ' + str((month_t1 - month_t0) / 60))
        
    lIdx += 2


        
    
        


