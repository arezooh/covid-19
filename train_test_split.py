import pandas as pd

# this code splits the historical data into two train and test datasets
# the output is two files for each dataset
def train_test_split(datasetPath):
	dataset = pd.read_csv(datasetPath)

	totalNumberOfDays = len(dataset['date of day t'].unique())
	totalNumberOfCounties = len(dataset['FIPS code'].unique())

	# last 14 days are for test dataset and the rest are for train dataset
	train_dataset = dataset.head(totalNumberOfCounties * (totalNumberOfDays-14))
	test_dataset = dataset.tail(14*totalNumberOfCounties)

	# strong datasets to files
	train_dataset.to_csv('train_dataset.csv')
	test_dataset.to_csv('test_dataset.csv')


def main():
	datasetPath = 'dataset_h=3.csv'
	train_test_split(datasetPath)


if __name__ == "__main__":
	main()
