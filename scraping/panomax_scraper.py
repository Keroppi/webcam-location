import urllib.request
from bs4 import BeautifulSoup
import string, sys, json, requests, re, datetime
from functools import reduce
from selenium import webdriver # To scrape websites built with Javascript
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, NoSuchElementException

today = str(datetime.date.today())

hdr = {'User-Agent': 'Mozilla/5.0'}
url = 'https://www.panomax.com/index_en.php'

GOOGLE_MAPS_API_KEY = 'AIzaSyBJR3_-E1p-hZP--SAAjB2ucYo4T_ufxS0'

req = urllib.request.Request(url, headers=hdr)
page = urllib.request.urlopen(req)
content = page.read()
soup = BeautifulSoup(content, "lxml")

scripts = soup.findAll('script')
script = scripts[16].text

s = '"instances":['
cam_start = script.find(s)
script = script[cam_start + len(s):]

s = '"cams":['
end = script.find(s)

cam_ids = script[:end - 2] # Contains cam_id, name
#print(cam_ids)
    
script = script[end + len(s):]

s = '// start'
end = script.find(s)

end = script.rfind(']', 0, end)
metadata = script[:end] # Contains lat/lng, country
#print(metadata)

# Get the webcam ids and names.
webcams = {}
while True:
    s = '"id":'
    start = cam_ids.find(s)
    
    if start == -1:
        break

    start = start + len(s)
    end = cam_ids.find(',', start)
    url_id = cam_ids[start:end]

    s = '"cam_id":'
    start = cam_ids.find(s, end)
    start = start + len(s)
    end = cam_ids.find(',', start)
    cam_id = cam_ids[start:end]

    s = '"name":"'
    start = cam_ids.find(s, end)
    start = start + len(s)
    end = cam_ids.find('"', start)
    
    name = cam_ids[start:end]
    name = bytes(name, 'ascii').decode('unicode-escape')
    name = name.replace('\\', '')
    webcams[cam_id] = [name, 0, 0, 0, 0, 0, url_id]

    cam_ids = cam_ids[end:]
        
    #print("%s %s %s" % (cam_id, name, url_id))

# Get latitude, longitude, and country.
countries = {'at':'Austria', 'de':'Germany', 'fr':'France', 'ch':'Switzerland', 'gr':'Greece', 'it':'Italy', 'gb':'Great Britain', 'il':'Israel', 'au':'Australia', 'fi':'Finland', 'null':'Ocean'}

while True:
    s = '"id":'
    start = metadata.find(s)
    
    if start == -1:
        break

    start = start + len(s)
    end = metadata.find(',', start)

    cam_id = metadata[start:end]

    s = '"latitude":"'
    
    start = metadata.find(s, end)        
    start = start + len(s)
    end = metadata.find('"', start)

    latitude = metadata[start:end] 

    s = '"longitude":"'
    start = metadata.find(s, end)
    start = start + len(s)
    end = metadata.find('"', start)

    longitude = metadata[start:end]     
    
    s = '"firstPano":"'
    start = metadata.find(s, end)
    start = start + len(s)
    end = metadata.find('"', start)

    firstPano = metadata[start:end]

    s = '"lastPano":"'
    start = metadata.find(s, end)
    start = start + len(s)
    end = metadata.find('"', start)

    lastPano = metadata[start:end]     

    s = '"country":'
    start = metadata.find(s, end)
    start = start + len(s)
    end = metadata.find(',', start)    

    country = metadata[start:end]
    country = country.replace('"', '')
    #print(cam_id + " " + country)
    
    country = countries[country]
    
    webcams[cam_id][1] = latitude
    webcams[cam_id][2] = longitude
    webcams[cam_id][3] = country
    webcams[cam_id][4] = firstPano
    webcams[cam_id][5] = lastPano
    
    metadata = metadata[end:]
    
    #print((cam_id, webcams[cam_id]))

# Sort by country.

cams_by_country = {'Austria':[], 'Germany':[], 'France':[], 'Switzerland':[], 'Greece':[], 'Italy':[], 'Great Britain':[], 'Israel':[], 'Australia':[], 'Finland':[], 'Ocean':[]}
for key in webcams:
    storage_id = key
    country = webcams[key][3]
    cam = [storage_id] + webcams[key][:3] + webcams[key][4:]

    #print(type(key))
    #print(key + " " + str(country))
    cams_by_country[country].append(cam)

# Output to file.

for key in cams_by_country:
    if key == 'Ocean': # Ignore Ocean, because it's a moving cruiseship.
        continue
    
    print(key)
    print('# Webcams: ' + str(len(cams_by_country[key])))

    for cam in cams_by_country[key]:
        print(cam[1])
        print('http://' + cam[6] + '.panomax.com')
        print(cam[2] + " " + cam[3])
        print(cam[0])
        print('')
    print('-----')
    print('')
    
    
    
    
    
