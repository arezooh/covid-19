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

countiesData_temporal = {}
countiesData_fix = {}

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

def binary_search(target_fips, target_date):
    global countiesData_temporal
    target = (target_fips, date.fromisoformat(target_date))

    l = 0
    r = len(countiesData_temporal)

    # Find first row of target county
    while (1):
        mid = (r - l) // 2
        fips = int(countiesData_temporal[l + mid]['county_fips'], 10)

        if (fips == target[0] and l + mid > 0 and int(countiesData_temporal[l + mid - 1]['county_fips'], 10) != target[0]):
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

def calculateGridData(counties):
    global countiesData_temporal, countiesData_fix
    confirmed = 0
    death = 0
    virusPressure = 0
    virusPressure_weightSum = 0
    meat_plants = 0
    social_distancing_visitation_grade = 0
    social_distancing_visitation_grade_weightSum = 0
    for county in counties:
        # index = binary_search(countiesData_temporal, county['fips'], (startDay + timedelta(days=i)).isoformat())
        index_temporal, index_fix = calculateIndex(county['fips'], (startDay + timedelta(days=i)).isoformat())
        if (index_temporal != -1):
            confirmed += round(float(countiesData_temporal[index_temporal]['confirmed']) * county['percent'])
            death += round(float(countiesData_temporal[index_temporal]['death']) * county['percent'])
            meat_plants += round(int(countiesData_fix[index_fix]['meat_plants'], 10) * county['percent'])
            virusPressure += float(countiesData_temporal[index_temporal]['virus-pressure']) * county['percent']
            virusPressure_weightSum += county['percent']
            social_distancing_visitation_grade += float(countiesData_temporal[index_temporal][ 'social-distancing-visitation-grade']) * county['percent']
            social_distancing_visitation_grade_weightSum += county['percent']

    if virusPressure_weightSum != 0:
        virusPressure /= virusPressure_weightSum
    if social_distancing_visitation_grade_weightSum != 0:
        social_distancing_visitation_grade /= social_distancing_visitation_grade_weightSum

    return [confirmed, round(virusPressure, 2), meat_plants, death, round(social_distancing_visitation_grade, 1)]

if __name__ == "__main__":
    gridIntersection = loadIntersection('map_intersection_1.json')
    countiesData_temporal = loadCounties('full-temporal-data.csv')
    countiesData_fix = loadCounties('full-fixed-data.csv')

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
                gridCell = calculateGridData(gridIntersection[x][y])
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
    # countiesData_temporal.clear()
    # imageArray.clear()

    # # ################################################################ init model
    # model = keras.Sequential()
    # # # Conv2D parameters: filters, kernel_size, activation, input_shape
    # model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    # model.add(tf.keras.layers.MaxPooling2D(2,2))
    # # model.add(keras.input(shape=(len(imageArray), len(imageArray[0]), len(imageArray[0][0]), len(imageArray[0][0][0]))))
    # # model.add(layers.Dense(len(imageArray) * len(imageArray[0]) * len(imageArray[0][0]) * len(imageArray[0][0][0]))))
    # # model.add()