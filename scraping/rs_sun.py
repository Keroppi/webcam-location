#!/srv/glusterfs/vli/.pyenv/shims/python

import string, sys, os, requests, urllib.request, json, multiprocessing as mp, datetime, time, traceback, socket, random, glob
from urllib.error import ContentTooShortError
from http.client import RemoteDisconnected
from urllib.request import URLError
from urllib.request import HTTPError
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from bs4 import BeautifulSoup

CLUSTER = False # run on cluster or local machine
SIZE = 'small' # 'large'

GOOGLE_MAPS_API_KEY = 'AIzaSyCEJkK4hEYYnRv4z6hL6n8A8VqfqJdspnY'

if CLUSTER:
    SGE_TASK_ID = int(os.environ.get('SGE_TASK_ID')) # Determines the month that gets downloaded. (1 -> January)
    baseLocation = '/srv/glusterfs/vli/data/roundshot/'
else:
    LOCAL_MONTH = 3
    baseLocation = '~/data/roundshot/'
    baseLocation = os.path.expanduser(baseLocation)

year = '2018'

rs_filename = '~/roundshot.txt'
rs_filename = os.path.expanduser(rs_filename)

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:55.0) Gecko/20100101 Firefox/55.0')]
urllib.request.install_opener(opener)

with open(rs_filename) as rs_file:
    lines = rs_file.read().splitlines()


def get_sun_info(country, name, year, month, day, html_rows, lat, lng):
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
    #if not (os.path.isfile(sun_file_str) and os.path.getsize(sun_file_str) > 0):
    if True:
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
        elif tds[0].text == '' or tds[0].text == '-': # No sunrise because the sun was up all day yesterday.
            local_sunrise_str = 'SUN DID NOT RISE'

            if tds[1].text.find(':') >= 0:
                local_sunset_str = tds[1].text.split()[0] + ':00'
            else:
                local_sunset_str = 'SUN DID NOT SET'

            sun_time = 'N/A'
            sun_time_seconds = '-1'
        elif tds[1].text == '' or tds[1].text == '-': # No sunset because the sun is up all day from now into tomorrow.
            local_sunset_str = 'SUN DID NOT SET'

            if tds[0].text.find(':') >= 0:
                local_sunrise_str = tds[1].text.split()[0] + ':00'
            else:
                local_sunrise_str = 'SUN DID NOT RISE'

            sun_time = 'N/A'
            sun_time_seconds = '-1'
        else:
            stripped_sunrise = tds[0].text.replace('-', '')
            stripped_sunset = tds[1].text.replace('-', '')

            local_sunrise_str = stripped_sunrise.split()[0] + ':00'
            local_sunset_str = stripped_sunset.split()[0] + ':00'
            sun_time = tds[2].text

            if sun_time == '' or sun_time == '-':
                sun_time = 'N/A'
                sun_time_seconds = '-1'
            else:
                sun_time_split = sun_time.split(':')
                sun_time_split.reverse()
                sun_time_seconds = str(sum([int(x) * 60**power for power, x in enumerate(sun_time_split)]))

                if len(sun_time_split) == 2: # less than 1 hour day length
                    sun_time = '00:' + sun_time
                elif len(sun_time_split) == 1: # less than 1 minute day length - UNLIKELY
                    sun_time = '00:00:' + sun_time

        # Get the time zone offset (for local time).
        sunrise_date = datetime.datetime.strptime(date, "%Y-%m-%d")
        sunset_date = datetime.datetime.strptime(date, "%Y-%m-%d")

        # Heuristic for 2 if statements below - not necessarily 100% accurate.
        # If sunrise is from 10 PM to 11:59 PM, it happens the day before.
        if int(local_sunrise_str.split(':')[0]) >= 20:
            sunrise_date += datetime.timedelta(days=-1)
        # If sunset is from midnight to 1:59 AM, it happens the next day.
        if int(local_sunset_str.split(':')[0]) < 2:
            sunset_date += datetime.timedelta(days=1)

        d = datetime.datetime.strptime(str(sunrise_date.date()) + ' ' + local_sunrise_str, "%Y-%m-%d %H:%M:%S")
        d1 = datetime.datetime.strptime(str(sunset_date.date()) + ' ' + local_sunset_str, "%Y-%m-%d %H:%M:%S")
        timestamp = time.mktime(d.timetuple())
        google_url = 'https://maps.googleapis.com/maps/api/timezone/json?location=' + str(lat) + ',' + \
                     str(lng) + '&timestamp=' + str(timestamp) + '&key=' + GOOGLE_MAPS_API_KEY

        retries = 0
        while True:
            with urllib.request.urlopen(google_url) as goog_url_obj:
                time_data = json.loads(goog_url_obj.read().decode())

                if time_data['status'] == 'OK':
                    break
                else:
                    if retries < 600:
                        time.sleep(2)  # Possibly rate limited.
                        retries += 1
                    else:
                        print('Exceed Google API quota - sleeping.')  # Exceed quota for the day.
                        sys.stdout.flush()
                        time.sleep(3600)  # 1 hour
                        retries = 0

        offset = time_data['dstOffset'] + time_data['rawOffset']  # seconds
        utc_d = d - datetime.timedelta(seconds=offset)

        # Requery time zone using estimate of UTC time.
        timestamp = time.mktime(utc_d.timetuple())
        google_url = 'https://maps.googleapis.com/maps/api/timezone/json?location=' + str(lat) + ',' + \
                     str(lng) + '&timestamp=' + str(timestamp) + '&key=' + GOOGLE_MAPS_API_KEY
        retries = 0
        while True:
            with urllib.request.urlopen(google_url) as goog_url_obj:
                time_data = json.loads(goog_url_obj.read().decode())

                if time_data['status'] == 'OK':
                    break
                else:
                    if retries < 600:
                        time.sleep(2)  # Possibly rate limited.
                        retries += 1
                    else:
                        print('Exceed Google API quota - sleeping.')  # Exceed quota for the day.
                        sys.stdout.flush()
                        time.sleep(3600)  # 1 hour
                        retries = 0

        offset = time_data['dstOffset'] + time_data['rawOffset'];  # seconds

        # NOTE: Ultimately converting local time to UTC is IMPOSSIBLE because of ambiguity.
        # e.g. If clocks roll back from 2 AM to 1 AM, then we see 1 AM twice - ambiguous!
        # But if sunrise and sunset are not close to DST changes, then it should be ok.

        utc_sunrise = d - datetime.timedelta(seconds=offset)
        utc_sunset = d1 - datetime.timedelta(seconds=offset)

        # Find UTC solar noon.
        # Mali is at 0 longitude and never observes DST, so it is ALWAYS UTC time there.
        mali_url = 'https://www.timeanddate.com/sun/@17.5707,0?month=' + str(utc_d.month) + '&year=' + str(utc_d.year)
        mali_page = urllib.request.urlopen(mali_url).read()
        mali_soup = BeautifulSoup(mali_page, "lxml")
        mali_table = mali_soup.find_all('table')[0]
        mali_html_rows = mali_table.find_all('tr')
        for start_idx, row in enumerate(mali_html_rows):
            header_cells = row.find_all('th')
            if len(header_cells) > 0:
                if header_cells[0].text.split()[0] == '1':
                    break
        mali_html_rows = mali_html_rows[start_idx:]  # row corresponding to 1st day of that month

        for row_idx in range(len(mali_html_rows)):
            headers = mali_html_rows[row_idx].find_all('th')

            if len(headers) > 0:
                if headers[0].text.split()[0] == str(utc_d.day):
                    break

        mali_tds = mali_html_rows[row_idx].find_all('td')
        mali_split = mali_tds[10].text.split()
        mali_solar_noon = mali_split[0]

        # Save UTC (first 2 rows), day length (HH:MM:SS), 
        # timezone offset (sec), local time, and day length (sec).
        with open(wrtPth + '/sun.txt', 'w') as sun_file:
            sun_file.write(str(utc_sunrise) + '\n')
            sun_file.write(str(utc_sunset) + '\n')
            sun_file.write(sun_time + '\n')
            sun_file.write(str(offset) + '\n')
            sun_file.write(str(sunrise_date.date()) + ' ' + local_sunrise_str + '\n')
            sun_file.write(str(sunset_date.date()) + ' ' + local_sunset_str + '\n')
            sun_file.write(sun_time_seconds + '\n')
            sun_file.write(str(utc_d.date()) + ' ' + mali_solar_noon)

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
            future_download = {executor.submit(get_sun_info, country, name, year, month, day, html_rows, lat, lng): day for day in range(1, numDays + 1)}
                
            for _ in as_completed(future_download):
                pass
        month_t1 = time.time()
        print('Month Time (sec): ' + str((month_t1 - month_t0)))
        
    lIdx += 2


        
    
        


