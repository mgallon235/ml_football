import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


# Return numeric columns only

def numeric_cols(dataset):
    numerics = dataset.select_dtypes(include= np.number).columns.tolist()
    return numerics


# Define a function to plot histograms
def plot_hist_mult(data, var1, var2, bins):
    #sns.set(style="ticks") # Set Seaborn style
    #sns.set(style="darkgrid")
    sns.set(style="dark")


    fig, axes = plt.subplots(var1, var2)

    for i, el in enumerate(list(data.columns.values)):
        ax = data[el].plot.hist(ax=axes.flatten()[i], bins=bins, color='skyblue', edgecolor='black')  # Add color and edgecolor
        ax.set_title(f'Histogram of {el}', fontsize=12)
        ax.set_xlabel(el, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)

    fig.set_size_inches(12, 10)
    plt.tight_layout()  # Improve spacing
    plt.show()


# Define a function to plot box plots
def plot_box_mult(data, var1, var2):
    sns.set(style="dark") # Set Seaborn style

    fig, axes = plt.subplots(var1, var2)

    for i, el in enumerate(list(data.columns.values)):
        ax = data.boxplot(el, ax=axes.flatten()[i], fontsize='large', patch_artist=True, boxprops={'facecolor': 'white', 'color': 'black'})  # Add color
        ax.set_title(f'Boxplot of {el}', fontsize=12)  # Add title
        ax.set_ylabel(f'{el} Value', fontsize=10)  # Label y-axis
        
         # Customize the color of the median line (percentile 50)
        median_line = ax.lines[4]  
        median_line.set_color('red') 

    fig.set_size_inches(12, 10)
    plt.tight_layout()  # Improve spacing
    plt.show()


    ######################## Summary Statitstics

    ### dataset functions ###

def pop_parameters(dataset):
    statistics = {}

    for i in dataset.columns:
        size = np.size(dataset[i])
        datatype = dataset[i].dtypes
        unique_values = dataset[i].unique().size
        mean = np.mean(dataset[i])
        stdv = np.std(dataset[i])
        min = dataset[i].min()
        per25 = dataset[i].quantile(0.25)
        median = dataset[i].quantile(0.50)
        per75 = dataset[i].quantile(0.75)
        max = dataset[i].max()
        IQRs = dataset[i].quantile(0.75) - dataset[i].quantile(0.25)
        lower_bound = (dataset[i].quantile(0.25)) - 1.5*(dataset[i].quantile(0.75) - dataset[i].quantile(0.25))
        upper_bound = (dataset[i].quantile(0.75)) + 1.5*(dataset[i].quantile(0.75) - dataset[i].quantile(0.25))

        statistics[i] = (size,datatype,unique_values,mean,stdv,min,per25,median,per75,max,IQRs,lower_bound,upper_bound)
    results = pd.DataFrame.from_dict(statistics,orient='index',columns=['size','datatype','unique_values','mean','stdv','min','per25',
                                                                        'median','per75','max','IQRs','lower_bound','upper_bound'])
    #results['lower_bound'] = np.where((results['lower_bound']<0) & (results['min']>=0),0,results['lower_bound'])
    return round(results,2)


'''Saving results to a dictionary'''

def pop_parameters_dict(dataset):
    statistics = {}

    for i in dataset.columns:
        size = np.size(dataset[i])
        datatype = dataset[i].dtypes
        unique_values = dataset[i].unique().size
        mean = np.mean(dataset[i])
        stdv = np.std(dataset[i])
        min = dataset[i].min()
        per25 = dataset[i].quantile(0.25)
        median = dataset[i].quantile(0.50)
        per75 = dataset[i].quantile(0.75)
        max = dataset[i].max()
        IQRs = dataset[i].quantile(0.75) - dataset[i].quantile(0.25)
        lower_bound = (dataset[i].quantile(0.25)) - 1.5*(dataset[i].quantile(0.75) - dataset[i].quantile(0.25))
        upper_bound = (dataset[i].quantile(0.75)) + 1.5*(dataset[i].quantile(0.75) - dataset[i].quantile(0.25))

        statistics[i] = (size,datatype,unique_values,mean,stdv,min,per25,median,per75,max,IQRs,lower_bound,upper_bound)
    results = pd.DataFrame.from_dict(statistics,orient='index',columns=['size','datatype','unique_values','mean','stdv','min','per25',
                                                                        'median','per75','max','IQRs','lower_bound','upper_bound'])
    #results['lower_bound'] = np.where((results['lower_bound']<0) & (results['min']>=0),0,results['lower_bound'])
    results =  round(results,2)
    results = results.to_dict('index')
    return results