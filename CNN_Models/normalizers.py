from math import log, exp, sqrt
from numpy import array, zeros, save, load, copyto

################################################################ Classes

class minMax_normalizer:

    def __init__(self):
        self.minV = -1
        self.maxV = -1

    def update(self, value):
        if self.minV == -1 and self.maxV == -1:
            self.minV = value
            self.maxV = value

        elif value < self.minV:
            self.minV = value

        elif value > self.maxV:
            self.maxV = value

    def normal(self, value):
        if (self.maxV > self.minV):
            return (value - self.minV) / (self.maxV - self.minV)
        else:
            return self.maxV

    def get_min_max(self):
        return (self.minV, self.maxV)

    def set_min_max(self, minV, maxV):
        self.minV = minV
        self.maxV = maxV

    def inverse_normal(self, value):
        return (value * (self.maxV - self.minV)) + self.minV

class standardizer:

    def __init__(self):
        self.sum = 0
        self.sum_deviation = 0
        self.count = 0
        self.mean = 0
        self.deviation = 0

    def update_mean(self, value):
        self.sum += value
        self.count += 1

    def calculate_mean(self):
        if (self.count != 0):
            self.mean = self.sum / self.count

    def update_deviation(self, value):
        self.sum_deviation += pow(value - self.mean, 2)

    def calculate_deviation(self):
        if (self.count != 0):
            self.deviation = sqrt(self.sum_deviation / self.count)

    def standardize(self, value):
        if (self.deviation == 0):
            return 0
        return (value - self.mean) / self.deviation

    def inverse_standardize(self, value):
        return (value * self.deviation) + self.mean

    def get_mean_deviation(self):
        return (self.mean, self.deviation)

    def set_mean_deviation(self, mean, deviation):
        self.mean = mean
        self.deviation = deviation

    def check(self, b):
        if (self.mean == 0 and self.deviation == 0):
            log('mean and deviation zero in b={0} | sum={1}, sum_deviation={2}, count={3}'.format(b, self.sum, self.sum_deviation, self.count))

class logarithmic_normalizer:

    def normal(self, value):
        return log(value + 1)

    def inverse_normal(self, value):
        return exp(value) - 1

################################################################ 

# get a 4D numpy array and normalize it
def normal_x(train, validation, final_test):
    data_shape = train.shape
    no_validation = validation.shape[0]
    no_final_test = final_test.shape[0]

    minMax_obj = minMax_normalizer()
    log_obj = logarithmic_normalizer()

    # update minMax_obj for longitude data
    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for a in range(data_shape[2]):
                minMax_obj.update(train[i][j][a][2])

    normal_train = zeros((data_shape[0], data_shape[1], data_shape[2], data_shape[3]))
    normal_validation = zeros((no_validation, data_shape[1], data_shape[2], data_shape[3]))
    normal_final_test = zeros((no_final_test, data_shape[1], data_shape[2], data_shape[3]))

    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for a in range(data_shape[2]):
                for b in range(data_shape[3]):
                    # normal other data with logarithm
                    if (b != 2):
                        normal_train[i][j][a][b] = log_obj.normal(train[i][j][a][b])
                        if (i < no_validation):
                            normal_validation[i][j][a][b] = log_obj.normal(validation[i][j][a][b])
                        if (i < no_final_test):
                            normal_final_test[i][j][a][b] = log_obj.normal(final_test[i][j][a][b])
                    # normal longitude data with minMax
                    else:
                        normal_train[i][j][a][b] = minMax_obj.normal(train[i][j][a][b])
                        if (i < no_validation):
                            normal_validation[i][j][a][b] = minMax_obj.normal(validation[i][j][a][b])
                        if (i < no_final_test):
                            normal_final_test[i][j][a][b] = minMax_obj.normal(final_test[i][j][a][b])

    return (normal_train, normal_validation, normal_final_test)

def normal_y(train, validation, final_test):
    data_shape = train.shape
    no_validation = validation.shape[0]
    no_final_test = final_test.shape[0]

    obj_normalizer = logarithmic_normalizer()

    normal_train = zeros((data_shape[0], data_shape[1], data_shape[2], 1))
    normal_validation = zeros((no_validation, data_shape[1], data_shape[2], 1))
    normal_final_test = zeros((no_final_test, data_shape[1], data_shape[2], 1))

    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for a in range(data_shape[2]):
                normal_train[i][j][a][0] = obj_normalizer.normal(train[i][j][a])
                if (i < no_validation):
                    normal_validation[i][j][a][0] = obj_normalizer.normal(validation[i][j][a])
                if (i < no_final_test):
                    normal_final_test[i][j][a][0] = obj_normalizer.normal(final_test[i][j][a])

    return (normal_train, normal_validation, normal_final_test)

def inverse_normal_y(normal_data):
    data_shape = normal_data.shape

    obj_normalizer = logarithmic_normalizer()

    data = zeros(data_shape)

    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for a in range(data_shape[2]):
                for b in range(data_shape[3]):
                    data[i][j][a][b] = (obj_normalizer.inverse_normal(normal_data[i][j][a][b]))

    return data

################################################################
