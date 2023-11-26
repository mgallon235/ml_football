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


## Plotting Distplots
def plot_hist_mult_h2(data, var1, var2, hue_var):
    sns.set(style="darkgrid")

    num_variables = min(var1 * var2, data.shape[1])  # Use the minimum of specified subplots and number of variables
    var1 = (num_variables + var2 - 1) // var2  # Adjust var1 based on the number of variables

    fig, axes = plt.subplots(var1, var2, sharey=False, figsize=(12, 10), squeeze=False, subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw={'hspace': 0.5, 'wspace': 0.5})

    for i, el in enumerate(list(data.columns.values[:num_variables])):
        row = i // var2
        col = i % var2

        # Plot density plot for each category using sns.kdeplot
        for j, category in enumerate(data[hue_var].unique()):
            subset = data[data[hue_var] == category]
            sns.kdeplot(data=subset[el], fill=True, alpha=0.5, label=category, ax=axes[row, col], color=sns.color_palette('husl')[j])

        axes[row, col].set_title(f'Density Plot of {el}', fontsize=12)
        axes[row, col].set_xlabel(el, fontsize=10)
        axes[row, col].set_ylabel('Density', fontsize=10)
        axes[row, col].legend()

    plt.show()

# Example usage
# Assuming 'data' is your DataFrame, 'var1', 'var2' are subplot dimensions, and 'hue_var' is the categorical variable to hue by.
# plot_hist_mult_h(data, var1=2, var2=2, hue_var='another_categorical_variable')


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

########## Plotting CEF
# Player main characteristics
# Function for plotting group by means
def groupby_table(dataset,M):
    #Plotting both perc and accum
    #Plot both mean  and median as lines
    fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8) = plt.subplots(1,8, figsize=(14,8))
    dataset.plot(x= M, y='id' , kind= 'bar', ax=ax1,fontsize=8)
    dataset.plot(x=M, y='height_cm' , kind= 'bar', ax=ax2,fontsize=8)
    dataset.plot(x=M, y='weight_kg', kind = 'bar',ax = ax3,fontsize=8)
    dataset.plot(x=M, y='overall' , kind= 'bar', ax=ax4,fontsize=8)
    dataset.plot(x=M, y='potential' , kind= 'bar', ax=ax5,fontsize=8)
    dataset.plot(x=M, y='international_reputation' , kind= 'bar', ax=ax6,fontsize=8)
    dataset.plot(x=M, y='league_level' , kind= 'bar', ax=ax7,fontsize=8)
    dataset.plot(x=M, y='club_jersey_number' , kind= 'bar', ax=ax8,fontsize=8)
    

    plt.xticks(rotation=90)
    plt.show()

## CEF MEDIAN
def groupby_table_med(dataset,M):
#Plotting both perc and accum
#Plot both mean  and median as lines
    fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8) = plt.subplots(1,8, figsize=(14,8))
    dataset.plot(x= M, y='id' , kind= 'bar', ax=ax1,fontsize=8,color = 'orange')
    dataset.plot(x=M, y='height_cm' , kind= 'bar', ax=ax2,fontsize=8,color = 'orange')
    dataset.plot(x=M, y='weight_kg', kind = 'bar',ax = ax3,fontsize=8,color = 'orange')
    dataset.plot(x=M, y='overall' , kind= 'bar', ax=ax4,fontsize=8,color = 'orange')
    dataset.plot(x=M, y='potential' , kind= 'bar', ax=ax5,fontsize=8,color = 'orange')
    dataset.plot(x=M, y='international_reputation' , kind= 'bar', ax=ax6,fontsize=8,color = 'orange')
    dataset.plot(x=M, y='league_level' , kind= 'bar', ax=ax7,fontsize=8,color = 'orange')
    dataset.plot(x=M, y='club_jersey_number' , kind= 'bar', ax=ax8,fontsize=8,color = 'orange')
    

    plt.xticks(rotation=90)
    plt.show()


    # Player Finance characteristics
# Function for plotting group by means
def groupby_table2(dataset,M):
    #Plotting both perc and accum
    #Plot both mean  and median as lines
    fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(14,8))
    dataset.plot(x= M, y='id' , kind= 'bar', ax=ax1,fontsize=8)
    dataset.plot(x=M, y='value_eur' , kind= 'bar', ax=ax2,fontsize=8)
    dataset.plot(x=M, y='wage_eur', kind = 'bar',ax = ax3,fontsize=8)
    plt.xticks(rotation=90)
    plt.show()