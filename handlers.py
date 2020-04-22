# base imports
import csv
import json

# self imports
import debug

class handler_csv:
    def __init__(self):
        debug.debug_print("CSV Handler is up", 1)

    # Get 2 attribute and their data types, then compare to each ohter
    def _isEqual(self, firstAttribute, secondAttribute, dataType):
        if dataType == 'int':
            try:
                return int(firstAttribute, 10) == int(secondAttribute, 10)  # ERROR-throw exception but works fine... idk
            except:
                debug.debug_print("ERROR, comparing: %s & %s" % (firstAttribute, secondAttribute), 2)
        else:
            return firstAttribute == secondAttribute

    def _loadData(self, csvFilename):
        csvData = []
        csvFileFieldnames = []
        with open(csvFilename) as csvFile:
            csvDriver = csv.DictReader(csvFile)
            csvFileFieldnames = csvDriver.fieldnames
            for row in csvDriver:
                csvData.append(row)
        return (csvData, csvFileFieldnames)

    # Function that merge two CSV files on 'commonColumn'
    def merge_csvFiles_addColumns(self, csvFilename1, csvFilename2, destinationFilename, commonColumnNameOnFile1, commonColumnNameOnFile2, commonColumnDataType='string'):
        # Get csvFiles data
        csvData1, csvFile1Fieldnames = self._loadData(csvFilename1)
        csvData2, csvFile2Fieldnames = self._loadData(csvFilename2)
        
        # Generate final fieldnames after merge and remove duplication of 'commonColumn'
        mergedDataFieldnames = csvFile1Fieldnames + csvFile2Fieldnames
        mergedDataFieldnames.remove(commonColumnNameOnFile2)

        # Merge and save data 
        with open(destinationFilename, 'w') as destinationFile:
            csvDriver = csv.DictWriter(destinationFile, fieldnames=mergedDataFieldnames)
            csvDriver.writeheader()

            for row in csvData1:
                # Find correspondingRow on csvData2
                found = False
                for item in csvData2:
                    if self._isEqual(item.get(commonColumnNameOnFile2), row.get(commonColumnNameOnFile1), commonColumnDataType):
                        correspondingRow = item
                        found = True
                        break

                if found == False:
                    debug.debug_print("county not found:\t%s:%s" % (commonColumnNameOnFile1, row.get(commonColumnNameOnFile1)), 3)
                    continue

                # Remove duplication of commonColumn and update row
                del correspondingRow[commonColumnNameOnFile2]
                row.update(correspondingRow)
                # Add row to file
                csvDriver.writerow(row)
        
        debug.debug_print("SUCCESS: merge completed", 2)

    # Function that merge rows of two CSV files
    def merge_csvFiles_addRows(self, csvFilename1, csvFilename2, destinationFilename):
        # Get csvFiles data
        csvData1, csvFile1Fieldnames = self._loadData(csvFilename1)
        csvData2, csvFile2Fieldnames = self._loadData(csvFilename2)

        if csvFile1Fieldnames != csvFile2Fieldnames:
            debug.debug_print("ERROR: mismatch in columns", 2)
            return

        with open(destinationFilename, 'w') as destinationFile:
            csvDriver = csv.DictWriter(destinationFile, fieldnames=csvFile1Fieldnames)
            csvDriver.writeheader()
            csvDriver.writerows(csvData1)
            csvDriver.writerows(csvData2)

        debug.debug_print("SUCCESS: merge completed", 2)

    # Load a CSV file, save interested fields known as 'fieldnames' to DestinationFilename, ignore others
    def simplify_csvFile(self, csvFilename, destinationFilename, fieldnames):
        # Get csvFile data
        csvData = []
        with open(csvFilename) as csvFile:
            csvDriver = csv.DictReader(csvFile)
            for row in csvDriver:
                csvData.append(row)

        with open(destinationFilename, 'w') as DestinationFile:
            csvDriver = csv.DictWriter(DestinationFile, fieldnames)
            csvDriver.writeheader()
            for row in csvData:
                csvDriver.writerow({k:row[k] for k in (fieldnames) if k in row})
        
        debug.debug_print("SUCCESS: simplify completed", 2)

class handler_json:
    data = []
    def __init__(self):
        debug.debug_print("JSON Handler is up", 1)

    def _loadData(self, jsonFilename):
        jsonMetaData = []
        with open(jsonFilename) as jsonFile:
            jsonMetaData = json.load(jsonFile)
        return jsonMetaData

    # Function that transform a JSON file to CSV file. Design for hospitalBedMetaData
    def transform_jsonToCsv_hospitalBedData(self, jsonFilename, csvFilename):
        jsonMetaData = []
        fieldnames = ['county_fips', 'countyName', 'stateName', 'beds', 'unoccupiedBeds']
        jsonMetaData = self._loadData(jsonFilename)

        jsonCountiesData = jsonMetaData['objects']['counties']['geometries']
        singleCountyData = {}

        with open(csvFilename, 'w') as csvFile:
            csvDriver = csv.DictWriter(csvFile, fieldnames=fieldnames)
            csvDriver.writeheader()
            for row in jsonCountiesData:
                singleCountyData = {fieldnames[k]:unicode(row['properties'][fieldnames[k]]).encode('utf-8') for k in range(1, len(fieldnames))}
                singleCountyData['county_fips'] = int(row['id'], 10)
                csvDriver.writerow(singleCountyData)
        
        debug.debug_print("SUCCESS: transform completed(hospitalBedData)", 2)

    # Function that transform a JSON file to CSV file. Design for socialDistancingBedMetaData
    def transform_jsonToCsv_socialDistancingData(self, jsonFilename, csvFilename):
        jsonMetaData = []
        countyFieldnames = ['stateFips', 'stateName', 'countyFips', 'countyName']
        dataFieldnames = ['date', 'totalGrade', 'visitationGrade', 'encountersGrade', 'travelDistanceGrade']
        jsonMetaData = self._loadData(jsonFilename)

        jsonCountiesData = jsonMetaData['hits']['hits']
        singleCountyData = {}

        with open(csvFilename, 'w') as csvFile:
            csvDriver = csv.DictWriter(csvFile, fieldnames=(countyFieldnames+dataFieldnames))
            csvDriver.writeheader()
            for county in jsonCountiesData:
                # set county data
                singleCountyData = {field:county['_source'][field] for field in countyFieldnames}
                # set Grade of county for each day, specified with 'date'
                for data in county['_source']['data']:
                    singleCountyData.update({field:data[field] for field in dataFieldnames})
                    csvDriver.writerow(singleCountyData)
        
        debug.debug_print("SUCCESS: transform completed(socialDistancingData)", 2)
