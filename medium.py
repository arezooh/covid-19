# base imports
import csv

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
            self.downloadHandler.get_socialDistancingData(fips, 'sd-state%02d.json' % (fips))
            self.jsonHandler.transform_jsonToCsv_socialDistancingData('sd-state%02d.json' % (fips), 'socialDistancing-s%02d.csv' % (fips))

if __name__ == "__main__":
    jsonHandler = handler_json()
    csvHandler = handler_csv()
    downloadHandler = extractor()

    # downloadHandler.get_socialDistancingData(2, 'sd-state%02d.json' % (2))
    # downloadHandler.get_confirmAndDeathData(_JSON_Directory_ + 'confirmAndDeath.json')
    # csvHandler.simplify_csvFile('csvFiles/latitude.csv', 'csvFiles/simple-lat.csv', ['county_fips', 'lat'])
    # csvHandler.simplify_csvFile('csvFiles/hospital-beds.csv', 'csvFiles/simple-hospital-beds.csv', ['county_fips', 'beds(per1000)', 'unoccupiedBeds(per1000)'])
    # csvHandler.merge_csvFiles_byColumn('csvFiles/dataAndLat.csv', 'csvFiles/simple-hospital-beds.csv', 'csvFiles/fixed-data.csv', 'county FIPS code', 'county_fips')
    # csvHandler.merge_csvFiles_byRow('csvFiles/socialDistancing-s01.csv', 'csvFiles/socialDistancing-s11.csv', 'csvFiles/socialDistancing.csv')
    # jsonHandler.transform_jsonToCsv_confirmAndDeathData(_JSON_Directory_ + 'confirmAndDeath.json', _CSV_Directory_ + 'confirmAndDeath.csv')
    # jsonHandler.transform_jsonToCsv_hospitalBedData(_JSON_Directory_ + 'hospital-beds.json', _CSV_Directory_ + 'hospital-beds.csv')
    # jsonHandler.transform_jsonToCsv_socialDistancingData(_JSON_Directory_ + 'sd-state01.json', _CSV_Directory_ + 'socialDistancing-s01.csv')
    # jsonHandler.transform_jsonToCsv_socialDistancingData(_JSON_Directory_ + 'sd-state02.json', _CSV_Directory_ + 'socialDistancing-s02.csv')

    # medium = mediumClass()
    # medium.generate_allSocialDistancingData('temp.csv')