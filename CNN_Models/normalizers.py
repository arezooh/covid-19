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

    normalizers = []
    for b in range(data_shape[3]):
        if (b >= 6 and ((b - 6) % 4 == 0 or (b - 6) % 4 == 1)):
            normalizers.append(standardizer())
        else:
            normalizers.append(minMax_normalizer())

    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for a in range(data_shape[2]):
                for b in range(data_shape[3]):
                    if (b >= 6 and ((b - 6) % 4 == 0 or (b - 6) % 4 == 1)):
                        normalizers[b].update_mean(train[i][j][a][b])
                    else:
                        normalizers[b].update(train[i][j][a][b])

    # calculate standardizers mean
    for b in range(6, data_shape[3], 4):
            normalizers[b].calculate_mean()
            normalizers[b + 1].calculate_mean()

    # update standardizers deviation
    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for a in range(data_shape[2]):
                for b in range(6, data_shape[3], 4):
                    normalizers[b].update_deviation(train[i][j][a][b])
                    normalizers[b + 1].update_deviation(train[i][j][a][b + 1])

    # calculate standardizers deviation
    for b in range(6, data_shape[3], 4):
            normalizers[b].calculate_deviation()
            normalizers[b + 1].calculate_deviation()

    normal_train = zeros((data_shape[0], data_shape[1], data_shape[2], data_shape[3]))
    normal_validation = zeros((no_validation, data_shape[1], data_shape[2], data_shape[3]))
    normal_final_test = zeros((no_final_test, data_shape[1], data_shape[2], data_shape[3]))

    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for a in range(data_shape[2]):
                for b in range(data_shape[3]):
                    if (b >= 6 and ((b - 6) % 4 == 0 or (b - 6) % 4 == 1)):
                        normal_train[i][j][a][b] = normalizers[b].standardize(train[i][j][a][b])
                        if (i < no_validation):
                            normal_validation[i][j][a][b] = normalizers[b].standardize(validation[i][j][a][b])
                        if (i < no_final_test):
                            normal_final_test[i][j][a][b] = normalizers[b].standardize(final_test[i][j][a][b])
                    else:
                        normal_train[i][j][a][b] = normalizers[b].normal(train[i][j][a][b])
                        if (i < no_validation):
                            normal_validation[i][j][a][b] = normalizers[b].normal(validation[i][j][a][b])
                        if (i < no_final_test):
                            normal_final_test[i][j][a][b] = normalizers[b].normal(final_test[i][j][a][b])

    # check deviation and mean
    for b in range(6, data_shape[3], 4):
        normalizers[b].check(b)
        normalizers[b + 1].check(b + 1)

    return (normal_train, normal_validation, normal_final_test)

def normal_y(train, validation, final_test):
    data_shape = train.shape
    no_validation = validation.shape[0]
    no_final_test = final_test.shape[0]

    obj_normalizer = standardizer()
    
    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for a in range(data_shape[2]):
                obj_normalizer.update_mean(train[i][j][a])

    # calculate standardizers mean
    obj_normalizer.calculate_mean()
    
    # update standardizers deviation
    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for a in range(data_shape[2]):
                obj_normalizer.update_deviation(train[i][j][a])
                
    # calculate standardizers deviation
    obj_normalizer.calculate_deviation()

    normal_train = zeros((data_shape[0], data_shape[1], data_shape[2], 1))
    normal_validation = zeros((no_validation, data_shape[1], data_shape[2], 1))
    normal_final_test = zeros((no_final_test, data_shape[1], data_shape[2], 1))

    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for a in range(data_shape[2]):
                normal_train[i][j][a][0] = obj_normalizer.standardize(train[i][j][a])
                if (i < no_validation):
                    normal_validation[i][j][a][0] = obj_normalizer.standardize(validation[i][j][a])
                if (i < no_final_test):
                    normal_final_test[i][j][a][0] = obj_normalizer.standardize(final_test[i][j][a])

    obj_normalizer.check(100)
    standard_mean, standard_deviation = obj_normalizer.get_mean_deviation()

    return (normal_train, normal_validation, normal_final_test, standard_mean, standard_deviation)

def inverse_normal_y(normal_data, standard_mean, standard_deviation):
    data_shape = normal_data.shape

    obj_normalizer = standardizer()
    obj_normalizer.set_mean_deviation(standard_mean, standard_deviation)

    data = zeros(data_shape)

    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for a in range(data_shape[2]):
                for b in range(data_shape[3]):
                    data[i][j][a][b] = (obj_normalizer.inverse_standardize(normal_data[i][j][a][b]))

    return data

################################################################
