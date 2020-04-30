# base imports
import requests
import json
import csv
import progressbar

# self imports
import debug

# defines
_CSV_Directory_ = './csvFiles/'
_JSON_Directory_ = './jsonFiles/'

# An object for downloading data online and save to drive
class extractor:
    def __init__(self):
        debug.debug_print("Extractor is up", 1)

    def get_socialDistancingData(self, stateFIPS, destinationFilename):
        jsonData = requests.get('https://covid19-scoreboard-api.unacastapis.com/api/search/covidcountyaggregates_v3?q=stateFips:%02d&size=4000' % (stateFIPS)).json()
        with open(_JSON_Directory_ + destinationFilename, 'w') as jsonFile:
            json.dump(jsonData, jsonFile)

        debug.debug_print("SUCCESS: data extracted (socialDistancingData on stateFIPS:%02d)" % (stateFIPS), 2)

    def get_confirmAndDeathData(self, destinationFilename):
        jsonData = requests.get('https://usafactsstatic.blob.core.windows.net/public/2020/coronavirus-timeline/allData.json').json()
        with open(_JSON_Directory_ + destinationFilename, 'w') as jsonFile:
            json.dump(jsonData, jsonFile)

        debug.debug_print("SUCCESS: data extracted (confirmAndDeathData)", 2)

    # A function that extract weather stations of all US counties
    def get_allStations(self, destinationFilename):
        countyData = []
        with open(_CSV_Directory_ + 'fixed-data.csv') as csvFile:
            csvDriver = csv.DictReader(csvFile)
            for row in csvDriver:
                countyData.append({'fips':row['county_fips'], 'name':row['county_name']})

        cookies = {
            'JSESSIONID': '54BF6C05EA3D53BF54F9349927E5CC62',
            '_ga': 'GA1.2.1721804174.1588253956',
            '_gid': 'GA1.2.850825076.1588253956',
            '_gat': '1',
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:68.0) Gecko/20100101 Firefox/68.0',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.ncdc.noaa.gov/cdo-web/datatools/selectlocation',
            'token': '0x2a',
            'X-Requested-With': 'XMLHttpRequest',
            'Connection': 'keep-alive',
        }

        fieldnames = [u'elevation', u'name', u'maxdate', u'datacoverage', u'longitude', u'latitude', u'elevationUnit', u'id', u'mindate']
        with open(_CSV_Directory_ + destinationFilename, 'w') as csvFile:
            csvDriver = csv.DictWriter(csvFile, fieldnames=fieldnames)
            csvDriver.writeheader()

            numberOfCounties = len(countyData)
            progressBar = progressbar.ProgressBar(maxval=numberOfCounties, widgets=[progressbar.Bar('#', '|', '|'), ' ', progressbar.Percentage()])
            progressBar.start()
            for i in range(numberOfCounties):
                params = (
                    ('limit', '1000'),
                    ('datasetid', 'GHCND'),
                    ('locationid', 'FIPS:%05d' % (int(countyData[i]['fips'], 10))),
                    ('sortfield', 'name'),
                )

                print('[*] sending request...')
                jsonData = requests.get('https://www.ncdc.noaa.gov/cdo-web/api/v2/stations', headers=headers, params=params, cookies=cookies).json()
                print('[*] request sent. Analyzing response')
                csvDriver.writerows(jsonData['results'])
                progressBar.update(i + 1)

            progressBar.finish()

        debug.debug_print("SUCCESS: data extracted (weather stations)", 2)

    def get_countyWeatherData(self, stationID, startDate, endDate, destinationFilename):
        csvData = requests.get('https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&dataTypes=&stations={0}&startDate={1}&endDate={2}&boundingBox=90,-180,-90,180'.format(stationID, startDate, endDate))
        with open(_CSV_Directory_ + destinationFilename, 'w') as csvFile:
            csvFile.write(csvData)

        debug.debug_print("SUCCESS: data extracted (weatherData of station:{0})".format(stationID), 2)


