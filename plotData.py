import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import random

''' this function plots time dependant data, for some random counties
each covariate is represented in one graph
number of counties to be included in graphs is "numberOfCountinesToBePlotted" '''
def plotData(numberOfCountinesToBePlotted=10):
    df = pd.read_csv('csvFiles/temporal-data.csv')

    # creating a list of all counties
    counties = df.county_fips.unique().tolist()

    # creating a list of time dependant covariates
    timeDependantCovariates = df.columns.values.tolist()
    timeDependantCovariates.remove('county_fips')
    timeDependantCovariates.remove('date')

    # choosing some random counties for plotting
    randomCounties = random.sample(counties, numberOfCountinesToBePlotted)
    ''' one county of new york must be in the list of counties, here we check this and if this situation doesn't have happened
    we add county with county_fips=36001 to the list '''
    if 36001 not in randomCounties:
        random_item_from_list = random.choice(randomCounties)
        randomCounties.remove(random_item_from_list)
        randomCounties.append(36001)
    
    # creating a list of colors, one color for each county
    number_of_colors = numberOfCountinesToBePlotted
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(number_of_colors)]

    # pdf is the output file
    pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")

    for covariate in timeDependantCovariates:
        # in each iteration newDataFrame includes values of one time dependant covariate for the counties selected before
        newDataFrame = pd.DataFrame()
        # completing newDataFrame happens in this loop
        for county in randomCounties:
            temp = df.loc[df['county_fips'] == county, ['date', covariate]]
            temp = temp.sort_values(by=['date'])
            temp.reset_index(drop=True, inplace=True)
            temp.rename(columns={covariate: (str(county))}, inplace=True)
            newDataFrame = pd.concat([newDataFrame, temp], axis=1)

        # removing duplicated columns
        newDataFrame = newDataFrame.loc[:,~newDataFrame.columns.duplicated()]
        #newDataFrame = newDataFrame.set_index('date')
        # next three lines moves date column to the first column
        dateCol = newDataFrame['date']
        newDataFrame.drop(labels=['date'], axis=1, inplace = True)
        newDataFrame.insert(0, 'date', dateCol)
        #max_value = newDataFrame.values.max()
        numberOfDays = len(newDataFrame.index)
        
        # plotting data
        ax = newDataFrame.plot.scatter(x='date', y=newDataFrame.columns[1], color=colors[0], 
                            figsize=(numberOfDays, 30), label=newDataFrame.columns[1], title=covariate, 
                            s=100, rot=90, fontsize=25)
        for cnt in range(2, 11):
            newDataFrame.plot.scatter(x='date', y=newDataFrame.columns[cnt], 
                            color=colors[cnt-1], label=newDataFrame.columns[cnt], 
                            s=100, rot=90, fontsize=25, ax=ax)
        # get figure of the graph
        fig = ax.get_figure()
        # save figure in the pdf file
        pdf.savefig(fig)
    pdf.close()


def main():
    numberOfCountiesToBePlotted = 10
    plotData(numberOfCountiesToBePlotted)


if __name__ == "__main__":
    main()