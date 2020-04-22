# base imports

# self imports
import debug
from handlers import handler_csv, handler_json
from extractor import extractor

# defines
_CSV_Directory_ = '../../covid-19/csvFiles/'
_JSON_Directory_ = './jsonFiles/'

if __name__ == "__main__":
    jsonHandler = handler_json()
    csvHandler = handler_csv()
    extractor = extractor()

    extractor.get_socialDistancingData(2, 'sd-state%02d.json' % (2))
    # csvHandler.merge e_csvFiles_byColumn('temp.csv', 'simpleCountyData.csv', 'merged.csv', 'county', 'CTYNAME')
    # csvHandler.simplify_csvFile('csvFiles/latitude.csv', 'csvFiles/simple-lat.csv', ['county_fips', 'lat'])
    # csvHandler.print_singleColumnCsvFile('csvFiles/simple-lat.csv', 'county_fips')
    # jsonHandler.transform_jsonToCsv_hospitalBedData('hospital-beds.json', 'csvFiles/hospital-beds.csv')
    # csvHandler.simplify_csvFile('csvFiles/hospital-beds.csv', 'csvFiles/simple-hospital-beds.csv', ['county_fips', 'beds(per1000)', 'unoccupiedBeds(per1000)'])
    # csvHandler.merge_csvFiles_byColumn('csvFiles/dataAndLat.csv', 'csvFiles/simple-hospital-beds.csv', 'csvFiles/fixed-data.csv', 'county FIPS code', 'county_fips')
    jsonHandler.transform_jsonToCsv_socialDistancingData('sd-state02.json', 'csvFiles/socialDistancing-s02.csv')
    # csvHandler.merge_csvFiles_byRow('csvFiles/socialDistancing-s01.csv', 'csvFiles/socialDistancing-s11.csv', 'csvFiles/socialDistancing.csv')