import urllib.request
from bs4 import BeautifulSoup
import string, sys, json, requests, re
from functools import reduce
from selenium import webdriver # To scrape websites built with Javascript
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, NoSuchElementException

hdr = {'User-Agent': 'Mozilla/5.0'}
url_begin = 'https://backend.roundshot.com/schema_list/'
url_end = '/frame.json'
countries = {'Switzerland':203, 'Germany':204, 'Norway':206, 'France':225, 'Finland':238, 'Denmark':239, 'Luxembourg':262, 'United Kingdom':236, 'Spain':375, 'North America':208, 'Australia':237, 'New Zealand':331, 'Russia':438} # Austria needs special processing

GOOGLE_MAPS_API_KEY = 'AIzaSyBJR3_-E1p-hZP--SAAjB2ucYo4T_ufxS0'

options = webdriver.ChromeOptions() # Need to run Headless Chrome at the same time - chromedriver
options.add_argument('headless')
options.add_argument('window-size=1280x720')
driver = webdriver.Chrome(chrome_options=options)

try:
    for country in countries.keys():
        print(country)    
        req = urllib.request.Request(url_begin + str(countries[country]) + url_end, headers=hdr)
        page = urllib.request.urlopen(req)
        content = page.read()
        soup = BeautifulSoup(content, "lxml")
    
        json_dict = json.loads(str(soup.text))
        categories = json_dict['categories'].values()
        webcams = reduce(lambda x,y: x+y, categories) # flatten list    
        print('# Webcams: ' + str(len(webcams)))
    
        for webcam in webcams:    
            webcam_url = webcam['link']
            location = webcam['name']
            location = location.replace(' ', '+')
            location = location.replace('Kachelmannwetter+-+', '') # Remove "weather"
            full_location = location + ',+' + country

            #if webcam_url != 'https://halter.roundshot.com/zwickyareal-10/':
            #    continue
            
            print(webcam['name'])
            print(webcam_url)
        
            driver.get(webcam_url)    
            loading_bar = WebDriverWait(driver, 99999).until(EC.presence_of_element_located((By.XPATH, "//div[@class='progress-void'][@style='width: 0px;']")))
            #driver.implicitly_wait(3) # Implicitly wait 3s for it to load.

            use_google = False
            
            # Get latitude and longitude.
            try:
                coordinates = driver.find_element_by_xpath("//*[contains(text(), '°') and contains(text(), '\"') and contains(text(), ';')]")
            except NoSuchElementException:
                use_google = True

            if not use_google:
                lat_long_str = coordinates.get_attribute('innerHTML')
                lat_long_str = lat_long_str.replace('°', '')
                lat_long_str = lat_long_str.replace('\"', '')
                lat_long_str = lat_long_str.replace('\'', '')
                lat = lat_long_str.split(';')[0]
                long = lat_long_str.split(';')[1]
                lat = re.sub("[^0-9]", " ", lat) # remove non-numerics
                long =  re.sub("[^0-9]", " ", long)
                lat = [float(x) for x in lat.split()]
                long = [float(x) for x in long.split()]
                lat = lat[0] + lat[1] / 60 + lat[2] / 3600
                long = long[0] + long[1] / 60 + long[2] / 3600
                print("%s %s" % (lat, long))
            else:
                api_req = 'https://maps.googleapis.com/maps/api/geocode/json?address=' + full_location + '&key=' + GOOGLE_MAPS_API_KEY
                api_response = requests.get(api_req)
                resp_json_payload = api_response.json()

                if len(resp_json_payload['results']) <= 0:            
                    api_req = 'https://maps.googleapis.com/maps/api/geocode/json?address=' + location + '&key=' + GOOGLE_MAPS_API_KEY # Try without the country
                    api_response = requests.get(api_req)
                    resp_json_payload = api_response.json()

                if len(resp_json_payload['results']) > 0:
                    lat_long = resp_json_payload['results'][0]['geometry']['location']
                    latitude = lat_long['lat']
                    longitude = lat_long['lng']                    
                    print("%s %s" % (latitude, longitude))
                else:
                    print("EXCEPTION - Lat/Long not found.")                       

            # Click the archive image to get storage id.
            archive_img = driver.find_element_by_xpath("//img[@data-ng-src='modules/sidebar/assets/img/archive.png']")

            no_archive_img = False
            tries = 0
            while True:
                try:
                    archive_img.click()
                    break            
                except WebDriverException:
                    driver.implicitly_wait(3)
                    tries = tries + 1

                    if tries > 50:
                        no_archive_img = True
                        break
                    
            #driver.get_screenshot_as_file('screenshot.png')
                    
            if no_archive_img: # Try another approach if no archive link.
                req1 = urllib.request.Request(webcam_url, headers=hdr)
                page1 = urllib.request.urlopen(req1)
                soup1 = BeautifulSoup(page1, "lxml")
                image_url = soup1.find_all('meta')[5].get('content')

                req2 = urllib.request.Request(image_url, headers=hdr)
                try:
                    page2 = urllib.request.urlopen(req2)
                    final_url = page2.geturl()
                    storage_id = final_url.split('/')[3]
                except Exception:
                    # Manually solve the ones that can't be found.
                    if webcam_url == 'https://longyearbyen.roundshot.com/':
                        storage_id = '53ac223476f774.62257077'
                    elif webcam_url == 'https://portlongyear.roundshot.com/':
                        storage_id = '53ac30d122b126.15751728'
                    elif webcam_url == 'https://senjahopen.roundshot.com/':
                        storage_id = '53ac3614053139.49019442'
                    else:
                        print('EXCEPTION - Cannot find storage id.')
            else:
                preview_img = WebDriverWait(driver, 99999).until(EC.presence_of_element_located((By.XPATH, "//img[@data-ng-click='switchToImage(previewedSeries[previewedKey].id)']")))
                storage_url = preview_img.get_attribute('src')
            
                while True:
                    if len(storage_url) == 0:
                        driver.implicitly_wait(3)
                        storage_url = preview_img.get_attribute('src')
                    else:
                        break
                
                storage_url = storage_url.replace('https://storage.roundshot.com/', '')
                storage_id = storage_url.split('/')[0]
                
            print(storage_id)
            print('')

            sys.stdout.flush()
            
        print('-----') # Delimit countries
        print('')
                
               
finally:
    driver.quit()    

