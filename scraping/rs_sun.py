#!/scratch_net/biwidl106/vli/.pyenv/shims/python

import string, sys, os, requests, urllib.request, json, multiprocessing as mp, datetime, time, traceback, socket, random, glob
from urllib.error import ContentTooShortError
from http.client import RemoteDisconnected
from urllib.request import URLError
from urllib.request import HTTPError
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from bs4 import BeautifulSoup

CLUSTER = True # run on cluster or local machine
SIZE = 'small' # 'large'

GOOGLE_MAPS_API_KEY = 'AIzaSyCEJkK4hEYYnRv4z6hL6n8A8VqfqJdspnY'

if CLUSTER:
    SGE_TASK_ID = int(os.environ.get('SGE_TASK_ID')) # Determines the month that gets downloaded. (1 -> January)
    baseLocation = '/srv/glusterfs/vli/data/roundshot/'
else:
    LOCAL_MONTH = 1 # January
    baseLocation = '~/data/roundshot/'
    baseLocation = os.path.expanduser(baseLocation)

year = '2017'

rs_filename = '~/roundshot.txt'
rs_filename = os.path.expanduser(rs_filename)

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:55.0) Gecko/20100101 Firefox/55.0')]
urllib.request.install_opener(opener)

with open(rs_filename) as rs_file:
    lines = rs_file.read().splitlines()


def get_sun_info(country, name, year, month, day, html_rows):
 try:
    day_num = day
    day = "{0:0=2d}".format(day)

    #print('Day: ' + day)
    #sys.stdout.flush()

    date = year + '-' + month + '-' + day
    wrtPth =  baseLocation + country + '/' + name + '/' + year + '/' + month + '/' + day
    wrtPthSmall = wrtPth + '/small'
    wrtPthLarge = wrtPth + '/large'
    
    try:
        os.makedirs(wrtPthSmall)
        os.makedirs(wrtPthLarge)
    except FileExistsError:
        pass

    
    # If file already exists and has size > 0, skip this.
    sun_file_str = wrtPth + '/sun.txt'
    if not (os.path.isfile(sun_file_str) and os.path.getsize(sun_file_str) > 0):
        # Find which row of the table corresponds to this day.
        for row_idx in range(len(html_rows)):
            headers = html_rows[row_idx].find_all('th')

            if len(headers) > 0:
                if headers[0].text.split()[0] == str(day_num):
                    break

        tds = html_rows[row_idx].find_all('td')

        if tds[0].text == 'Down all day':
            local_sunrise_str = 'NO SUN'
            local_sunset_str = 'NO SUN'
            sun_time = '00:00:00'
            sun_time_seconds = '0'
        elif tds[0].text == 'Up all day':
            local_sunrise_str = 'SUN ALL DAY'
            local_sunset_str = 'SUN ALL DAY'
            sun_time = '24:00:00'
            sun_time_seconds = str(24 * 60 * 60)
        elif tds[0].text == '': # No sunrise because the sun was up all day yesterday.
            local_sunrise_str = 'SUN DID NOT RISE'

            if tds[1].text.find(':') >= 0:
                local_sunset_str = tds[1].text.split()[0] + ':00'

            sun_time = 'N/A'
            sun_time_seconds = '-1'
        elif tds[1].text == '':
            local_sunset_str = 'SUN DID NOT SET'

            if tds[0].text.find(':') >= 0:
                local_sunrise_str = tds[1].text.split()[0] + ':00'

            sun_time = 'N/A'
            sun_time_seconds = '-1'
        else:
            local_sunrise_str = tds[0].text.split()[0] + ':00'
            local_sunset_str = tds[1].text.split()[0] + ':00'
            sun_time = tds[2].text

            sun_time_split = sun_time.split(':')
            sun_time_split.reverse()
            sun_time_seconds = str(sum([int(x) * 60**power for power, x in enumerate(sun_time_split)]))

            if len(sun_time_split) == 2: # less than 1 hour day length
                sun_time = '00:' + sun_time
            elif len(sun_time_split) == 1: # less than 1 minute day length - UNLIKELY
                sun_time = '00:00:' + sun_time

        # Save UTC (first 2 rows), day length (HH:MM:SS), 
        # timezone offset (sec), local time, and day length (sec).
        with open(wrtPth + '/sun.txt', 'w') as sun_file:
            sun_file.write('UTC SUNRISE UNKNOWN' + '\n')
            sun_file.write('UTC SUNSET UNKNOWN' + '\n')
            sun_file.write(sun_time + '\n')
            sun_file.write('TIME ZONE / DST OFFSET UNKNOWN' + '\n')
            sun_file.write(local_sunrise_str + '\n')
            sun_file.write(local_sunset_str + '\n')
            sun_file.write(sun_time_seconds + '\n')

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
        name = line
        name = name.replace('/', '-')  # Otherwise the / will create a sub-directory

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

        #with open(locWrtPth + 'location.txt', 'w+') as loc_file:
        #    loc_file.write(lat_long + '\n')
            
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

        sun_url = 'https://www.timeanddate.com/sun/@' + str(lat) + ',' + str(lng) + '?month=' + month + '&year=' + year
        #print(sun_url)
        page = urllib.request.urlopen(sun_url).read()
        soup = BeautifulSoup(page, "lxml")
        table = soup.find_all('table')[0]
        html_rows = table.find_all('tr')

        for start_idx, row in enumerate(html_rows):
            header_cells = row.find_all('th')
            if len(header_cells) > 0:
                if header_cells[0].text.split()[0] == '1':
                    break

        html_rows = html_rows[start_idx:] # row corresponding to 1st day of that month

        with ThreadPoolExecutor(numDays) as executor:               
            future_download = {executor.submit(get_sun_info, country, name, year, month, day, html_rows): day for day in range(1, numDays + 1)}
                
            for _ in as_completed(future_download):
                pass
        month_t1 = time.time()
        print('Month Time (sec): ' + str((month_t1 - month_t0)))
        
    lIdx += 2


        
    
        


