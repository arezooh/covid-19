import json
import csv
from datetime import date
from datetime import timedelta
import datetime
import time
import pandas as pd

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

_CSV_Directory_ = ''
_JSON_Directory_ = ''

startDay = date.fromisoformat('2020-01-22')
endDay = date.fromisoformat('2020-05-08')
dayLen = (endDay - startDay).days
dataTrain = []
dataTest = []
hashCounties = [-1] * 78031     #78030 is biggest county fips

input_shape = [0, 0, 0, 0]

def loadIntersection(jsonFilename):
    jsonMetaData = []
    with open(_JSON_Directory_ + jsonFilename) as jsonFile:
        jsonMetaData = json.load(jsonFile)
    return jsonMetaData

def loadCounties(csvFilename):
    csvData = []
    with open(_CSV_Directory_ + csvFilename) as csvFile:
        csvDriver = csv.DictReader(csvFile)
        for row in csvDriver:
            csvData.append(row)
    return csvData

# Compare two date. This function implemented for specefic use, getting baseDay object and counter, then compare it to targetDay in our Data format(e.g. 05/18/20)
def dayComp(baseDay, dayCounter, targetDay):
    day1 = (baseDay + timedelta(days=dayCounter)).isoformat()
    day2 = datetime.datetime.strptime(targetDay, '%m/%d/%y').strftime('%Y-%m-%d')
    if (day1 == day2):
        return True
    return False

def fromIsotoDataFormat(day):
    return day.strftime('%m/%d/%y')

def init_hashCounties():
    counties = loadCounties('full-data-county-fips.csv')
    for i in range(len(counties)):
        hashCounties[int(counties[i]['county_fips'], 10)] = i

def binary_search(countiesData, target_fips, target_date):
    target = (target_fips, date.fromisoformat(target_date))

    l = 0
    r = len(countiesData)

    # Find first row of target county
    while (1):
        mid = (r - l) // 2
        fips = int(countiesData[l + mid]['county_fips'], 10)

        if (fips == target[0] and l + mid > 0 and int(countiesData[l + mid - 1]['county_fips'], 10) != target[0]):
            l = l + mid
            r = l + 1000
            break

        elif (fips >= target[0]):
            r = l + mid

        else:
            l = l + mid + 1

        if (r == l):
            return -1

    target_daysFromStart = (target[1] - startDay).days
    if (target_daysFromStart <= dayLen):
        return l + target_daysFromStart
    else:
        return -1

def calculateIndex(target_fips, target_date):
    target = (target_fips, date.fromisoformat(target_date))

    target_daysFromStart = (target[1] - startDay).days
    target_countiesFromStart = hashCounties[target[0]]

    if (target_daysFromStart <= dayLen and target_countiesFromStart != -1):
        index = target_countiesFromStart * (dayLen + 1) + target_daysFromStart
        return (index, target_countiesFromStart)
    else:
        return (-1, target_countiesFromStart)

def calculateGridData(counties, countiesData):
    confirmed = 0
    array_virusPressure = []
    for county in counties:
        # index = binary_search(countiesData, county['fips'], (startDay + timedelta(days=i)).isoformat())
        index_temporal, index_fix = calculateIndex(county['fips'], (startDay + timedelta(days=i)).isoformat())
        if (index_temporal != -1):
            confirmed += round(float(countiesData[index_temporal]['confirmed']) * county['percent'])     
            array_virusPressure.append((float(countiesData[index_temporal]['virus-pressure']), county['percent']))

    virusPressure = calculateWeightedAverage(array_virusPressure)
    return [confirmed, virusPressure]

def calculateWeightedAverage(array):
    sum_value = 0
    sum_weight = 0

    for i in array:
        sum_value += i[0] * i[1]
        sum_weight += i[1]

    try:
        average = sum_value / sum_weight
        return round(average, 2)
    except:
        return 0

if __name__ == "__main__":
    gridIntersection = loadIntersection('map_intersection_1.json')
    countiesData = loadCounties('full-temporal-data.csv')

    init_hashCounties()

    ################################################################ creating image array(CNN input) ### Binary Search

    # each row on imageArray include image data on day i
    imageArray = []

    start_time = time.time()
    for i in range(dayLen):
        grid = []
        for x in range(len(gridIntersection)):
            gridRow = []
            for y in range(len(gridIntersection[x])):
                gridCell = calculateGridData(gridIntersection[x][y], countiesData)
                gridRow.append(gridCell)
            grid.append(gridRow)
        imageArray.append(grid)

    # Show data
    for i in range(len(imageArray)):
        print("day " + str(i))
        for x in range(len(imageArray[i])):
            for y in range(len(imageArray[i][x])):
                print(imageArray[i][x][y], end='')
            print('')
        print('')

    print('\t|Execution time: {0}'.format(time.time() - start_time))

    # ################################################################ split imageArray into train Data(dataTrain) and test Data(dataTest)

    # input_shape = [dayLen - 14, len(gridIntersection), len(gridIntersection[0]), 1]
    # dataTrain = imageArray[:-14]
    # dataTest = imageArray[-28:]

    # # x_dataTrain = dataTrain[:-14]
    # # y_dataTrain = dataTrain[14:]

    # # x_dataTest = dataTest[:-14]
    # # y_dataTest = dataTest[14:]

    # # Clear memory
    # gridIntersection.clear()
    # countiesData.clear()
    # imageArray.clear()

    # # ################################################################ init model
    # model = keras.Sequential()
    # # # Conv2D parameters: filters, kernel_size, activation, input_shape
    # model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    # model.add(tf.keras.layers.MaxPooling2D(2,2))
    # # model.add(keras.input(shape=(len(imageArray), len(imageArray[0]), len(imageArray[0][0]), len(imageArray[0][0][0]))))
    # # model.add(layers.Dense(len(imageArray) * len(imageArray[0]) * len(imageArray[0][0]) * len(imageArray[0][0][0]))))
    # # model.add()