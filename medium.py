# base imports
import csv
import os

# self imports
import debug
from handlers import handler_csv, handler_json
from extractor import extractor

# defines
_CSV_Directory_ = './csvFiles/'
_JSON_Directory_ = './jsonFiles/'

# Last step, don't need to complete it now
class mediumClass:
    jsonHandler = handler_json()
    csvHandler = handler_csv()
    downloadHandler = extractor()
    def __init__(self):
        
        debug.debug_print("Medium Class is up", 1)

    def generate_allSocialDistancingData(self, destinationFilename):
        statesData = self.csvHandler._loadData('states.csv')[0]
        for state in statesData:
            fips = int(state['state-fips'], 10)
            self.downloadHandler.get_socialDistancingData(fips, 'temp.json')
            # First step, create socialDistancing.csv file
            if state == statesData[0]:
                self.jsonHandler.transform_jsonToCsv_socialDistancingData('temp.json', 'socialDistancing.csv')
            # Other steps, merge new data to socialDistancing.csv file
            else:
                self.jsonHandler.transform_jsonToCsv_socialDistancingData('temp.json', 'temp.csv')
                self.csvHandler.merge_csvFiles_addRows('socialDistancing.csv', 'temp.csv', 'socialDistancing.csv')