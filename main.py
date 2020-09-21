#!/usr/bin/python3

# self imports
import debug
import handlers
import extractor
import medium

if __name__ == "__main__":
    # downloadHandler.get_socialDistancingData(2, 'sd-state%02d.json' % (2))
    # downloadHandler.get_confirmAndDeathData( + 'confirmAndDeath.json')
    # csvHandler.simplify_csvFile('csvFiles/latitude.csv', 'csvFiles/simple-lat.csv', ['county_fips', 'lat'])
    # csvHandler.simplify_csvFile('csvFiles/hospital-beds.csv', 'csvFiles/simple-hospital-beds.csv', ['county_fips', 'beds(per1000)', 'unoccupiedBeds(per1000)'])
    # csvHandler.merge_csvFiles_addRows('csvFiles/socialDistancing-s01.csv', 'csvFiles/socialDistancing-s11.csv', 'csvFiles/socialDistancing.csv')
    # jsonHandler.transform_jsonToCsv_confirmAndDeathData( + 'confirmAndDeath.json',  + 'confirmAndDeath.csv')
    # jsonHandler.transform_jsonToCsv_hospitalBedData( + 'hospital-beds.json',  + 'hospital-beds.csv')
    # jsonHandler.transform_jsonToCsv_socialDistancingData( + 'sd-state01.json',  + 'socialDistancing-s01.csv')
    # jsonHandler.transform_jsonToCsv_socialDistancingData('sd-state02.json',  + 'socialDistancing-s02.csv')

    # medium = mediumClass()
    # medium.generate_allSocialDistancingData('temp.csv')

    # jsonHandler.transform_jsonToCsv_confirmAndDeathData('confirmAndDeath.json', 'temp-confirmAndDeath.csv')
    # downloadHandler.get_allStations('stations.csv')
    # downloadHandler.get_countyWeatherData('USW00093228', '2020-04-19', '2020-04-28', 'test.csv') #https://www.ncdc.noaa.gov/cdo-web/datasets/GHCND/stations/GHCND:USW00093228/detail


    #       |--Use these two lines to get all stations--|
    # mediumObject = medium.mediumClass()
    # mediumObject.downloadHandler.get_allStations('stations.csv')
    #       |--|


    #       |--Use this line to test merging two temporalDataFile: 'confirmAndDeath.csv' and 'socialDistancing.csv'--|
    # mediumObject = medium.mediumClass()
    # mediumObject.csvHandler.merge_csvFiles_addColumns('confirmAndDeath.csv', 'socialDistancing.csv', 'temporal-data.csv', ['countyFIPS', 'date'], ['countyFips', 'date'], ['totalGrade', 'visitationGrade', 'encountersGrade', 'travelDistanceGrade'])
    #       |--|

    # mediumObject = medium.mediumClass()
    # mediumObject.downloadHandler.get_countyWeatherData('1001', 'USW00093228', '2020-05-02', '2020-05-04', 'test.csv')
    # mediumObject.generate_allWeatherData('2020-05-04', '2020-05-04')

    mediumObject = medium.mediumClass()
    mediumObject.downloadHandler.get_airlines()