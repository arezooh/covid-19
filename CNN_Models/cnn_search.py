import json
import csv
from datetime import date
from datetime import timedelta
import datetime
import time
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, MaxPooling2D, Dropout

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from numpy import array

from math import log2, floor

_CSV_Directory_ = ''
_JSON_Directory_ = ''

startDay = datetime.datetime.strptime('2020-01-22', '%Y-%m-%d')
endDay = datetime.datetime.strptime('2020-05-08', '%Y-%m-%d')
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
    counties = loadCounties('./full-data-county-fips.csv')
    for i in range(len(counties)):
        hashCounties[int(counties[i]['county_fips'], 10)] = i

def binary_search(target_fips, target_date):
    global countiesData_temporal
    target = (target_fips, datetime.datetime.strptime(target_date, '%Y-%m-%d'))

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
    target = (target_fips, datetime.datetime.strptime(target_date, '%Y-%m-%dT%H:%M:%S'))

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
    population = 0
    area = 0
    population_density = 0
    longitude = 0
    longitude_sum = 0
    social_distancing_travel_distance_grade = 0
    social_distancing_travel_distance_grade_weightSum = 0
    houses = 0
    houses_density = 0
    for county in counties:
        index_temporal, index_fix = calculateIndex(county['fips'], (startDay + timedelta(days=i)).isoformat())
        if (index_temporal != -1):
            confirmed += round(float(countiesData_temporal[index_temporal]['confirmed']) * county['percent'])
            death += round(float(countiesData_temporal[index_temporal]['death']) * county['percent'])
            meat_plants += round(int(countiesData_fix[index_fix]['meat_plants'], 10) * county['percent'])
            virusPressure += float(countiesData_temporal[index_temporal]['virus-pressure']) * county['percent']
            virusPressure_weightSum += county['percent']
            social_distancing_visitation_grade += float(countiesData_temporal[index_temporal][ 'social-distancing-visitation-grade']) * county['percent']
            social_distancing_visitation_grade_weightSum += county['percent']
            population += round(int(countiesData_fix[index_fix]['total_population'], 10) * county['percent'])
            area += float(countiesData_fix[index_fix]['area']) * county['percent']
            longitude += float(countiesData_fix[index_fix]['longitude'])
            longitude_sum += 1
            social_distancing_travel_distance_grade += float(countiesData_temporal[index_temporal]['social-distancing-travel-distance-grade']) * county['percent']
            social_distancing_travel_distance_grade_weightSum += county['percent']
            houses += float(countiesData_fix[index_fix]['houses_density']) * float(countiesData_fix[index_fix]['area']) * county['percent']

    if virusPressure_weightSum != 0:
        virusPressure /= virusPressure_weightSum
    if social_distancing_visitation_grade_weightSum != 0:
        social_distancing_visitation_grade /= social_distancing_visitation_grade_weightSum
    if area != 0:
        population_density = round(population / area, 2)
        houses_density = round(houses / area, 2)
    if longitude_sum != 0:
        longitude /= longitude_sum
    if social_distancing_travel_distance_grade_weightSum != 0:
        social_distancing_travel_distance_grade /= social_distancing_travel_distance_grade_weightSum

    return [confirmed, round(virusPressure, 2), meat_plants, death, round(social_distancing_visitation_grade, 1), population_density, population, round(longitude, 3), round(social_distancing_travel_distance_grade, 1), houses_density]

def init_days():
    global startDay
    global endDay
    global dayLen
    startDay = datetime.datetime.strptime(countiesData_temporal[0]['date'], '%m/%d/%y')
    endDay = startDay
    
    for row in countiesData_temporal:
        day = datetime.datetime.strptime(row['date'], '%m/%d/%y')
        if day > endDay:
            endDay = day
            dayLen = (endDay - startDay).days

        elif day == startDay and row != countiesData_temporal[0]:
            break

def split_d4Datas(imageArray, data_index):
    output = []
    for i in range(len(imageArray)):
        output.append([imageArray[i][data_index]])

    return output

# time_mainStart = time.time()

gridIntersection = loadIntersection('./map_intersection_square.json')
countiesData_temporal = loadCounties('./full-temporal-data.csv')
countiesData_fix = loadCounties('./full-fixed-data.csv')

init_hashCounties()
init_days()

################################################################ creating image array(CNN input) ### Binary Search

print('\t|--start creating image...')

# time_imageCreation = time.time()

# each row on imageArray include image data on day i
imageArray = []

for i in range(dayLen):
    grid = []
    for x in range(len(gridIntersection)):
        gridRow = []
        for y in range(len(gridIntersection[x])):
            gridCell = calculateGridData(gridIntersection[x][y])
            gridRow.append(gridCell)
        grid.append(gridRow)
    imageArray.append(grid)

# # Show data
# for i in range(len(imageArray)):
#     print("day " + str(i))
#     for x in range(len(imageArray[i])):
#         for y in range(len(imageArray[i][x])):
#             print(imageArray[i][x][y], end='')
#         print('')
#     print('')

print('\t|--SUCCESS: image created')

################################################################ normalize data

print('\t|--start normalizing data...')

# time_imageNormalization = time.time()

imageNormal = []
shape_imageArray = array(imageArray).shape

imageArray = array(imageArray).reshape(shape_imageArray[0] * shape_imageArray[1] * shape_imageArray[2], shape_imageArray[3])

normalizeObject_f0 = MinMaxScaler()
normalizeObject_f1 = MinMaxScaler()
normalizeObject_f2 = MinMaxScaler()
normalizeObject_f3 = MinMaxScaler()
normalizeObject_f4 = MinMaxScaler()
normalizeObject_f5 = MinMaxScaler()
normalizeObject_f6 = MinMaxScaler()
normalizeObject_f7 = MinMaxScaler()
normalizeObject_f8 = MinMaxScaler()
normalizeObject_f9 = MinMaxScaler()

imageArray_f0 = split_d4Datas(imageArray, 0)
imageArray_f1 = split_d4Datas(imageArray, 1)
imageArray_f2 = split_d4Datas(imageArray, 2)
imageArray_f3 = split_d4Datas(imageArray, 3)
imageArray_f4 = split_d4Datas(imageArray, 4)
imageArray_f5 = split_d4Datas(imageArray, 5)
imageArray_f6 = split_d4Datas(imageArray, 6)
imageArray_f7 = split_d4Datas(imageArray, 7)
imageArray_f8 = split_d4Datas(imageArray, 8)
imageArray_f9 = split_d4Datas(imageArray, 9)

imageArray_f0 = normalizeObject_f0.fit_transform(imageArray_f0)
imageArray_f1 = normalizeObject_f1.fit_transform(imageArray_f1)
imageArray_f2 = normalizeObject_f2.fit_transform(imageArray_f2)
imageArray_f3 = normalizeObject_f3.fit_transform(imageArray_f3)
imageArray_f4 = normalizeObject_f4.fit_transform(imageArray_f4)
imageArray_f5 = normalizeObject_f5.fit_transform(imageArray_f5)
imageArray_f6 = normalizeObject_f6.fit_transform(imageArray_f6)
imageArray_f7 = normalizeObject_f7.fit_transform(imageArray_f7)
imageArray_f8 = normalizeObject_f8.fit_transform(imageArray_f8)
imageArray_f9 = normalizeObject_f9.fit_transform(imageArray_f9)

for i in range(len(imageArray)):
    imageNormal.append([imageArray_f0[i][0], imageArray_f1[i][0], imageArray_f2[i][0],
                      imageArray_f3[i][0], imageArray_f4[i][0], imageArray_f5[i][0],
                      imageArray_f6[i][0], imageArray_f7[i][0], imageArray_f8[i][0], imageArray_f9[i][0]])

imageNormal = array(imageNormal)
imageNormal = imageNormal.reshape(shape_imageArray[0], shape_imageArray[1], shape_imageArray[2], shape_imageArray[3])
    
# time_lap = time.time()

# # Show data
# for i in range(len(imageNormal)):
#     print("day " + str(i))
#     for x in range(len(imageNormal[i])):
#         for y in range(len(imageNormal[i][x])):
#             print(imageNormal[i][x][y], end='')
#         print('')
#     print('')

print('\t|--SUCCESS: data normalized')

################################################################ split imageArray into train Data(dataTrain) and test Data(dataTest)

print('\t|--start spliting data into train, validation and test...')

# dataTrain = imageNormal[:-14]
# dataTest = imageNormal[-28:]

data_shape = (shape_imageArray[1], shape_imageArray[2], shape_imageArray[3])

x_dataTrain = imageNormal[:-28][:-14]
y_dataTrain = imageNormal[:-28][14:]
y_dataTrain_final = []
for i in range(len(y_dataTrain)):
    grid = []
    for j in range(data_shape[0]):
        gridRow = []
        for k in range(data_shape[1]):
            gridCell = [y_dataTrain[i][j][k][0]]
            gridRow.append(gridCell)
        grid.append(gridRow)
    y_dataTrain_final.append(grid)

y_dataTrain = array(y_dataTrain_final)

x_dataValidation = imageNormal[-42:-14][:-14]
y_dataValidation = imageNormal[-42:-14][14:]
y_dataValidation_final = []
for i in range(len(y_dataValidation)):
    grid = []
    for j in range(data_shape[0]):
        gridRow = []
        for k in range(data_shape[1]):
            gridCell = [y_dataValidation[i][j][k][0]]
            gridRow.append(gridCell)
        grid.append(gridRow)
    y_dataValidation_final.append(grid)

y_dataValidation = array(y_dataValidation_final)

x_dataTest = imageNormal[-28:][:-14]
y_dataTest = imageNormal[-28:][14:]
y_dataTest_final = []
for i in range(len(y_dataTest)):
    grid = []
    for j in range(data_shape[0]):
        gridRow = []
        for k in range(data_shape[1]):
            gridCell = [y_dataTest[i][j][k][0]]
            gridRow.append(gridCell)
        grid.append(gridRow)
    y_dataTest_final.append(grid)

y_dataTest = array(y_dataTest_final)

# Clear memory
gridIntersection.clear()
countiesData_temporal.clear()
countiesData_fix.clear()
# imageArray.clear()
# imageNormal.clear()

print('\t|--SUCCESS: data splited into train, validation and test')

################################################################ print execution time
    
# time_endTime = time.time()

# print('\t|Image creation time: {0}'.format(time_imageNormalization - time_imageCreation))
# print('\t|Image normalization time: {0}'.format(time_lap - time_imageNormalization))
# print('\t|full execution time: {0}'.format(time_endTime - time_mainStart))

def create_model(inputSize, hiddenDropout, visibleDropout, noBlocks, noDenseLayer, increaseFilters):
    noFilters = 64
    model = keras.Sequential()

    # Layers before first block
    model.add(tf.keras.layers.Conv2D(filters=noFilters, kernel_size = (3,3), padding='same', activation='relu', input_shape=(inputSize, inputSize, 10)))
    if (visibleDropout != 0):
        model.add(Dropout(visibleDropout))

    # layers in Blocks
    for i in range(noBlocks):
        if (increaseFilters == 1):
            noFilters = 64 * pow(2, i)
        model.add(Conv2D(filters=noFilters, kernel_size = (3,3), padding='same', activation="relu"))
        model.add(Conv2D(filters=noFilters, kernel_size = (3,3), padding='same', activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        if (hiddenDropout != 0):
            model.add(Dropout(hiddenDropout))

    # Layers after last block
    for i in range(noDenseLayer - 1):
        model.add(Dense(512,activation="relu"))
    # Last layer
    model.add(Dense(1,activation="relu"))

    model.compile('adam', 'mean_squared_error', metrics=['accuracy'])
    # model.compile(loss=keras.losses.poisson, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    # model.compile(optimizer='adam', loss=tf.keras.losses.Poisson())
    return model

# This function expand the image, to get output size equal to input size
def pad_data(data, input_size):
    n = input_size // 2

    padded_data = list(data)
    for j in range(n):
        padded_data.insert(0, padded_data[0])
        padded_data.append(padded_data[-1])

    for i in range(len(padded_data)):
        padded_data[i] = list(padded_data[i])
        for j in range(n):
            padded_data[i].insert(0, padded_data[i][0])
            padded_data[i].append(padded_data[i][-1])
    return array(padded_data)

# This function extract windows with "input_size" size from image, train model with the windows data
def train_data(model, x_train, y_train, x_validation, y_validation, NO_epochs, input_size):
    data_shape = x_train.shape
    y_noData = y_train.shape[-1]
    for i in range(data_shape[1] - input_size + 1):
        for j in range(data_shape[2] - input_size + 1):
            subX_trian = x_train[0:data_shape[0], i:i+input_size, j:j+input_size, 0:data_shape[3]]
            subY_train = y_train[0:data_shape[0], i:i+input_size, j:j+input_size, 0:y_noData]

            subX_validation = x_validation[0:data_shape[0], i:i+input_size, j:j+input_size, 0:data_shape[3]]
            subY_validation = y_validation[0:data_shape[0], i:i+input_size, j:j+input_size, 0:y_noData]

            model.fit(subX_trian, subY_train, batch_size=32, epochs=NO_epochs, verbose=1, validation_data=(subX_validation, subY_validation))
            # model.fit(x_dataTrain, y_dataTrain, batch_size=32, epochs=25, verbose=1, validation_data=(x_dataValidation, y_dataValidation))

# This function extract windows with "input_size" size from image, evaluate model with the windows data
def evaluate_data(model, x_test, y_test, input_size):
    data_shape = x_test.shape
    y_noData = y_test.shape[-1]
    sum_loss = 0
    sum_acc = 0
    total = 0

    for i in range(data_shape[1] - input_size + 1):
        for j in range(data_shape[2] - input_size + 1):
            subX_test = x_test[0:data_shape[0], i:i+input_size, j:j+input_size, 0:data_shape[3]]
            subY_test = y_test[0:data_shape[0], i:i+input_size, j:j+input_size, 0:y_noData]

            score = model.evaluate(subX_test, subY_test, verbose=0)
            sum_loss += score[0]
            sum_acc += score[1]

            total += 1

    return (sum_loss / total, sum_acc / total)

# Use this function to log states of code, helps to find bugs
def log(str):
    t = datetime.datetime.now().isoformat()
    with open('log', 'a') as logFile:
        logFile.write('[{0}] {1}\n'.format(t, str))

################################################################ systematic search for find best model
# We change 5 parameters to find best model (for now, we can't change number of blocks(NO_blocks))
# input_size = [3, 5, 15, 25] where image size is 300*300
# hidden_dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
# visible_dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

# NO_dense_layer = [1, 2, 3]
# increase_filters = [0, 1]
################################################################

print('\t|--Phase of testing models started...')

p1 = [3, 5, 15, 25]
p2 = [0, 0.2, 0.3, 0.4]
p3 = [0, 0.2, 0.3, 0.4]
p4 = [1, 2, 3]
p5 = [0, 1]

best_result = (0, 0)
best_parameters = (-1, -1, -1, -1, -1)


for input_size in p1:
    for hidden_dropout in p2:
        for visible_dropout in p3:
            for NO_dense_layer in p4:
                for increase_filters in p5:
                    NO_blocks = floor(log2(input_size))
                    # print(input_size, hidden_dropout, visible_dropout, NO_blocks, NO_dense_layer, increase_filters)
                    # log this state
                    log('create_model({0}, {1}, {2}, {3}, {4}, {5})'.format(input_size, hidden_dropout, visible_dropout, NO_blocks, NO_dense_layer, increase_filters))
                    #
                    model = create_model(input_size, hidden_dropout, visible_dropout, NO_blocks, NO_dense_layer, increase_filters)
                    train_data(model, pad_data(x_dataTrain, input_size), pad_data(y_dataTrain, input_size), pad_data(x_dataValidation, input_size), pad_data(y_dataValidation, input_size), 2, input_size)
                    result = evaluate_data(model, pad_data(x_dataTest, input_size), pad_data(y_dataTest, input_size), input_size)
                    log('result | LOSS:{0} | ACC:{1}'.format(result[0], result[1]))
                    # update best_result if the accuray was better
                    if (result[1] > best_result[1]):
                        best_result = result
                        best_parameters = (input_size, hidden_dropout, visible_dropout, NO_dense_layer, increase_filters)
                        