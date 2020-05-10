import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import random

''' this function plots time dependant data, for some random counties
each covariate is represented in one graph
number of counties to be included in graphs is "numberOfCountinesToBePlotted" '''
def plotData(numberOfCountinesToBePlotted=10):
    # reading data
    timeDependantData = pd.read_csv('csvFiles/temporal-data.csv')
    independantOfTimeData = pd.read_csv('csvFiles/fixed-data.csv')

    # creating a list of time dependant covariates
    timeDependantCovariates = timeDependantData.columns.values.tolist()
    timeDependantCovariates.remove('county_fips')
    timeDependantCovariates.remove('date')

    # creating a new dataframe to be used for plotting
    independantOfTimeData = independantOfTimeData[['county_fips', 'county_name']]
    data = pd.merge(timeDependantData ,independantOfTimeData)

    # choosing some random counties for plotting
    random_fips_codes = random.sample(data['county_fips'].unique().tolist(), numberOfCountinesToBePlotted-1)
    nameOfCounties = [independantOfTimeData.loc[independantOfTimeData['county_fips'] == i, 'county_name'].iloc[0] for i in random_fips_codes]
    counties = dict(zip(random_fips_codes, nameOfCounties))

    # pdf is the output file
    pdf = matplotlib.backends.backend_pdf.PdfPages('output.pdf')

    dates = data['date'].unique()
    figs = plt.figure()

    # this outer loop goes through time dependant covariates, we access covariates with their position in 'timeDependantCovariates' list
    for cnt in range(0, len(timeDependantCovariates), 4):
        # creating a figure that will contain 4 graphs, size is for an A4 page
        fig = plt.figure(figsize=(21, 29.7))
        # with subplot_number we set the position of graph in page
        subplot_number = 411
        # this inner loop, each time goes through four time dependant covariates to plot them in one figure and save them in one page of pdf file
        for i in range(cnt, cnt+4):
            # checking for not to be out of band
            if i < len(timeDependantCovariates):
                plt.subplot(subplot_number)
                # new york must be in all of grphs, so we fisrt plot information about new york in grphs
                plt.plot(dates, data.loc[data['county_name'].str.contains('New York'), timeDependantCovariates[i]].tolist(), label='New York')
                # and here we plot information about other countines
                for fips_code in random_fips_codes:
                    plt.plot(dates, data.loc[data['county_fips']==fips_code, timeDependantCovariates[i]].tolist(), label=counties[fips_code])
                plt.ylabel(timeDependantCovariates[i])
                plt.xlabel('date')
                plt.xticks(rotation=60)
                plt.legend()
                subplot_number += 1
        # saving the figure in pdf file
        pdf.savefig(fig)
    pdf.close()


def main():
    numberOfCountiesToBePlotted = 10
    plotData(numberOfCountiesToBePlotted)


if __name__ == "__main__":
    main()