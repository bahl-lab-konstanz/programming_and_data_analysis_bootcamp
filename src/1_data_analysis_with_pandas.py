# DATA ANALYSIS WITH PANDAS
# For this section we will need an external library, pandas. So, we will use an "import" statement, that provides
# our script with the requested package from our conda environment. For convenience, we will use an abbreviation using
# the "as" keyword to give the imported library an alias
import pandas as pd

# we also import a bunch of other libraries and utilities that we will use along this section
from pathlib import Path
import numpy as np
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
df = pd.read_csv(path)

# %% print information
print(df.info())

# %% Let's peek at the data. Show the first 10 lines of the dataframe
print(df.head())

# We are not 100% happy with the result. In fact, we have an additional row, named "Unnamed: 0". Furthermore, we notice
# that that column contains just increasing numbers. In our output we can see that we already have an unnamed set of
# increasing numbers identifying the rows (the leftmost column of values).
# This happens because any dataframe needs an index, i.e. a non-ambiguous way to refer to rows.

# %% To correctly extract the information on the index, we need to provide pandas with more information about how we want
# the data to be loaded:
df = pd.read_csv(path, index_col=0, header=0)

# %% Let's incorporate it in a function, that also shows the first 10 lines of the dataframe
def import_csv(path, show_preview=False):
    # load data
    df = pd.read_csv(path, index_col=0, header=0)

    # show preview
    if show_preview:
        print(df.head())

    # return result
    return df


# %% And test the function
df = import_csv(path, show_preview=True)


# %% pandas provides an easy way to access basic statistical characteristics of our data. In particular dataframes have
# built-in methods "min()", "max()", "mean()", "sum()", "corr()", and "describe()" which gives a summary.
# Print basic statistics of the columns "x_position" and "y_position"
print("X POSITION")
print(df["x_position"].describe())

print("Y POSITION")
print(df["y_position"].describe())


# %% plot the trajectory of the fish

# select the first 100 movements of the fish
df_start = df.iloc[:100]

# show trajectory
plt.figure()
plt.plot(df_start["x_position"], df_start["y_position"], marker="o")
plt.show()


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
n_dark_side = len(df[df["side"] == -1])
n_edge = len(df[df["side"] == 0])
n_light_side = len(df[df["side"] == 1])

# organize data to plot them
bin_labels = [-1, 0, 1]
hist_values = [n_dark_side, n_edge, n_light_side]

# show bar plot
plt.figure()
plt.bar(bin_labels, hist_values)
plt.show()


# %% Plot histogram of the current fish for its x_position

# compute histogram values and bins
hist_values, bin_edges = np.histogram(df["x_position"])

# import function for centering bins
from src.utils.useful_functions import center_bins_hist

# center bins
bin_centers = center_bins_hist(bin_edges)

# plot result
plt.figure()
plt.plot(bin_centers, hist_values)
plt.show()


# %% we like the result, so let's embed this code in a function
def plot_histogram(df, column_name="x_position", axs=None):
    # import necessary support function
    from src.utils.useful_functions import center_bins_hist

    # compute histogram quantities
    hist_values, bin_edges = np.histogram(df[column_name])

    # center the bins
    bin_centers = center_bins_hist(bin_edges)

    # check if we already have a target window for plotting, if not create one
    if axs is None:
        fig, axs = plt.subplots((1, 1))

    # plot
    axs.plot(bin_centers, hist_values)


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


# %% Test the function all data in a single dataframe
dir_path = Path("./data")
df_all_fish = import_csv_from_dir(dir_path, show_info=True)


# %% Show distribution for all fish

# get list of fish IDs
fish_id_list = df_all_fish["fish_ID"].unique()

# initialize window for plots
fig, axs = plt.subplots(nrows=len(fish_id_list), ncols=1)


i_fish = 0
# loop over fish
for fish_id in fish_id_list:
    # filter dataframe using only data from current fish
    df_fish = df_all_fish[df_all_fish["fish_ID"] == fish_id]

    # select where to plot
    plot_section = axs[i_fish]

    # select what to show on the x axis
    plot_section.set_xlim(-4.5, 4.5)

    # draw vertical line on zero
    plot_section.axvline(x=0, color="gray", linestyle='--')

    # plot using the function we designed before
    plot_histogram(df_fish, column_name="x_position", axs=plot_section)

    # update fish index
    i_fish += 1

# show result
fig.show()


# %% study brightness preference

# select thresholds for preference
dark_threshold = -0.5
light_threshold = 0.5

# implement a definition for undecidedness
def is_undecided(peak_of_distribution):
    return dark_threshold < peak_of_distribution < light_threshold

# implement a definition for light preference
def prefers_light(peak_of_distribution):
    return peak_of_distribution > light_threshold

# implement a definition for dark preference
def prefers_dark(peak_of_distribution):
    return peak_of_distribution < dark_threshold


# %% check the preference of all our fish

# initialize empty list of undecided fish (we want to store them for later)
undecided_fish_list = []

# loop over fish
for fish_id in fish_id_list:
    # filter dataframe using only data from current fish
    df_fish = df_all_fish[df_all_fish["fish_ID"] == fish_id]

    # compute histogram quantities
    hist_values, bin_edges = np.histogram(df_fish["x_position"])

    # center the bins
    bin_centers = center_bins_hist(bin_edges)

    # select the mode
    i_max = np.argmax(hist_values)

    # select corresponding bin
    bin_max = bin_centers[i_max]

    if prefers_dark(bin_max):
        # log if the fish is undecided
        print(f"fish {fish_id} prefers DARKNESS. Its mode x_position is {bin_max}")
    elif prefers_light(bin_max):
        # log if the fish is undecided
        print(f"fish {fish_id} prefers LIGHT. Its mode x_position is {bin_max}")
    elif is_undecided(bin_max):
        # log if the fish is undecided
        print(f"fish {fish_id} is UNDECIDED. Its mode x_position is {bin_max}")
        # store the ID
        undecided_fish_list.append(fish_id)


# %% EXTRA
# To find the preference we computed the mode of x_position. Is there a more reliable way to obtain this information?
# How would the result change? Code it.


# %% check how undecided are the fish

# implement function to check when the fish is changing side
def compute_changing_side(df):
    # initialize array of zeros. We will update the values to 1 whenever the fish changes side
    changing_side_array = np.zeros(len(df))

    for i_row in range(1, len(df)):
        if df.iloc[i_row]["side"] * df.iloc[i_row-1]["side"] <= 0:
            changing_side_array[i_row] = 1

    # add a column to the df containing the information about changing side
    df["changing_side"] = changing_side_array

    # return updated dataframe
    return df

# %% quantify undecidedness of the fish

# loop over undecided fish
for fish_id in undecided_fish_list:
    # filter dataframe using only data from current fish
    df_fish = df_all_fish[df_all_fish["fish_ID"] == fish_id]

    # filter dataframe to recordings close to the edge
    df_fish = df_fish[np.logical_and(df_fish["x_position"] > dark_threshold,
                                     df_fish["x_position"] < light_threshold)]

    # add side column
    compute_side_fast(df_fish)

    # add changing_side column to the dataframe
    compute_changing_side(df_fish)

    # compute percentage of times in which the fish changed its mind
    percentage_changing_side = df_fish["changing_side"].sum() / len(df_fish) * 100

    # log result
    print(f"fish {fish_id} changed side in {percentage_changing_side:.02f}% of the cases")







