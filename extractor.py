# base imports
import requests
import json

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


