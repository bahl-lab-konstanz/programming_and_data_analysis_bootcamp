# DATA ANALYSIS WITH PANDAS AND NUMPY
# For this section we will need two external packages, pandas and numpy. So, we will use two "import" statements, that
# provide our script with the requested packages from our conda environment. For convenience, we will use an abbreviation
# using the "as" keyword to give the imported library an alias
import pandas as pd
import numpy as np

# we also import a bunch of other libraries and utilities that we will use along this section
from pathlib import Path
import matplotlib.pyplot as plt


# %% Introduction
# When working with tabular data, such as data stored in spreadsheets or databases, pandas is the perfect tool.
# It allows to easily explore, clean, and process data. In pandas, a data table is called a DataFrame.
# The name "pandas" stands for "Python Data Analysis" (or Panel Data according to some sources).

# %% First of all we have to provide the path of the location where we saved the data. Here we are using the relative
# path (relative to the present script). You can also provide it in form of absolute path. We start by loading one chunk
# of our dataset
path = "./data/fish_00.csv"

# %% We then procede loading the data and showing the overall information. Load data in a dataframe structure


# %% print information


# %% Let's peek at the data. Show the first 10 lines of the dataframe


# We are not 100% happy with the result. In fact, we have an additional row, named "Unnamed: 0". Furthermore, we notice
# that that column contains just increasing numbers. In our output we can see that we already have an unnamed set of
# increasing numbers identifying the rows (the leftmost column of values).
# This happens because any dataframe needs an index, i.e. a non-ambiguous way to refer to rows.

# %% To correctly extract the information on the index, we need to provide pandas with more information about how we want
# the data to be loaded:


# %% EXERCISE 0
# Let's incorporate it in a function, that also shows the first 10 lines of the dataframe
def import_csv(path, show_preview=False):
    # load data


    # show preview


    # return result



# %% EXERCISE 1
# And test the function



# %% EXERCISE 2
# pandas provides an easy way to access basic statistical characteristics of our data. In particular dataframes have
# built-in methods "min()", "max()", "mean()", "sum()", "corr()", and "describe()" which gives a summary.
# Print basic statistics of the columns "x_position" and "y_position"





# %% plot the trajectory of the fish

# select the first 100 movements of the fish


# show trajectory



# %% Implement a function to add a column to the dataframe, telling us on which side the fish is at the end of each swim

# slow implementation
def compute_side(df):
    # initialize array of zeros
    side = np.zeros(len(df))

    # loop over rows in the dataframe
    for i_row, row in df.iterrows():

        # check side and assign a label (-1: dark, 0: edge, 1: light)
        if row["x_position"] < 0:
            side[i_row] = -1
        elif row["x_position"] > 0:
            side[i_row] = 1
        else:
            side[i_row] = 0

    # add column to the dataframe
    df["side"] = side

# import package for timing (profiling) your code
import time

# execute code with manual profiling
time_start = time.time()
compute_side(df)
time_end = time.time()

# log result
print(f"The SLOW implementation of compute_side takes {time_end - time_start} s")


# %% Faster implementation
def compute_side_fast(df):
    # compute side
    side = np.sign(df["x_position"])

    # add column to the dataframe
    df["side"] = side

# execute code with manual profiling
time_start = time.time()
compute_side_fast(df)
time_end = time.time()

# log result
print(f"The FAST implementation of compute_side takes {time_end - time_start} s")


# %% Plot histogram of the presence of the fish on each side

# count events on each side


# organize data to plot them


# show bar plot



# %% Plot histogram of the current fish for its x_position

# compute histogram values and bins


# import function for centering bins


# center bins


# plot result



# %% EXERCISE 3
# We like the result, so let's embed this code in a function
def plot_histogram(df, column_name="x_position", axs=None):
    # import necessary support function


    # compute histogram quantities


    # center the bins


    # check if we already have a target window for plotting, if not create one


    # plot


# PRO-TIP: move useful functions in specific files grouping by scope of use. This way you will be able to easily find,
# import and reuse them throughout your project


# %% The dataset we just used come from a single fish. For our analysis we need to put together data from different
# individuals
def import_csv_from_dir(dir_path, show_info=False):
    # initialize empty list
    data_list = []

    # loop over all files and directories in dir_path
    for fish_dataset_path in dir_path.glob("*.csv"):
        # check if it is a file
        if fish_dataset_path.is_dir():
            continue

        # load data in a temporary structure
        df_temp = import_csv(fish_dataset_path)

        # add new batch to the list of datasets
        data_list.append(df_temp)

    # merge all the dataframes in data_list
    df = pd.concat(data_list, axis=0)

    # show information
    if show_info:
        print(df.info())

    # return result
    return df


# %% Test the function, merging all data into a single dataframe



# %% Show distribution for all fish together



# %% Show distribution for each fish

# get list of fish IDs


# initialize window for plots


# intialize a fish index, we will use it in a bit
i_fish = 0
# loop over fish_ids

    # filter dataframe using only data from current fish


    # select where to plot


    # select what to show on the x axis


    # draw vertical line on zero


    # plot using the function we designed before


    # update fish index


# show result



# %% Study brightness preference

# select thresholds for preference
dark_threshold = -0.5
light_threshold = 0.5

# implement a definition for undecidedness
def is_undecided(peak_of_distribution):


# implement a definition for light preference
def prefers_light(peak_of_distribution):


# implement a definition for dark preference
def prefers_dark(peak_of_distribution):



# %% EXERCISE 4
# Check the preference of all our fish

# initialize empty list of undecided fish (we want to store them for later)


# loop over fish_ids

    # filter dataframe using only data from current fish


    # compute histogram quantities


    # center the bins


    # select the mode


    # select corresponding bin


    # log if the fish is undecided


    # log if the fish is undecided


    # log and store the ID if the fish is undecided




# %% EXTRA: numpy
# To find the preference we computed the mode of x_position. Is there a more reliable way to obtain this information
# from the histograms? How would the result change? Code it.

# loop over fish_ids

    # filter dataframe using only data from current fish


    # compute histogram quantities


    # center the bins


    # merge hist values in dark/undecided/light areas and for each area compute the width of its x-range


    # define list with results. Preference computed as relative number of recordings in each area


    # find preference


    # log preference



# %% EXTRA: df.diff()
# Find how much time each fish spent in each area
# loop over fish_ids

    # filter dataframe using only data from current fish


    # compute time between recordings and assign it to new column


    # compute sum of time spent in each area


    # log results




# %% EXERCISE 5
# Check how undecided are the fish

# implement function to check when the fish is changing side
def compute_changing_side(df):
    # initialize array of zeros. We will update the values to 1 whenever the fish changes side


    # loop over rows in the dataframe

        # check if side has changed between two consecutive recordings

            # in case this is true, update the array with information on changing_side


    # add a column to the df containing the information about changing side



# %% EXERCISE 6
# Quantify undecidedness of the fish

# loop over undecided fish

    # filter dataframe using only data from current fish


    # filter dataframe to recordings close to the edge



    # add side column


    # add changing_side column to the dataframe


    # compute percentage of times in which the fish changed its mind


    # log result



# %% EXTRA: groupby
# Use groupby to efficiently compute the mean and standard deviation of y_position to check that fish has no vertical
# preference

# compute mean of all columns grouping data by fish_ID


# compute std of all columns grouping data by fish_ID


# extract mean and std of y_position for each fish



# log result for each fish



# PRO-TIP: sometimes mean and std are not enough to describe your data. They are only good metrics if your data has a
# gaussian distribution


