import pandas as pd

def correlatinosWithTarget(dataset):

	# Computing correlation of each Covariate with Target
	corr_matrix = dataset.corr()
	covariates = corr_matrix["Target"].sort_values(ascending=False)
	# This list contains name of covariates, ordered by correlations
	nameOfCovariates = list(covariates.index.values.tolist())
	# newDataFrame is the new data frame that it's columns are ordered by correlations that obtained
	newDataFrame = pd.DataFrame()
	for name in nameOfCovariates:
		newDataFrame[name] = dataset[name]
	#newDataFrame['date of day t'] = dataset['date of day t']

	return newDataFrame, nameOfCovariates


def main():
	path = ('dataset_h=3.csv')
	# Reading dataset
	dataset = pd.read_csv(path)
	newDataFrame, correlations = correlatinosWithTarget(dataset)
	# Store new data frame to a csv file
	newDataFrame.to_csv('new_' + inputDataset, mode='w', index=False)


if __name__ == "__main__":
	main()
