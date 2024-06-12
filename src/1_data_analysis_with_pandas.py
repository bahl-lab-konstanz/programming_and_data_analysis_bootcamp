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

# %% EXERCISE 0
# Let's incorporate it in a function, that also shows the first 10 lines of the dataframe
def import_csv(path, show_preview=False):
    # load data
    df = pd.read_csv(path, index_col=0, header=0)

    # show preview
    if show_preview:
        print(df.head())

    # return result
    return df


# %% EXERCISE 1
# And test the function
df = import_csv(path, show_preview=True)


# %% EXERCISE 2
# pandas provides an easy way to access basic statistical characteristics of our data. In particular dataframes have
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


# %% EXERCISE 3
# We like the result, so let's embed this code in a function
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


# %% Test the function all data in a single dataframe
dir_path = Path("./data")
df_all_fish = import_csv_from_dir(dir_path, show_info=True)


# %% Show distribution for all fish together
plot_histogram(df, column_name="x_position", axs=None)


# %% Show distribution for each fish

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


# %% Study brightness preference

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


# %% EXERCISE
# Check the preference of all our fish

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


# %% EXTRA: numpy
# To find the preference we computed the mode of x_position. Is there a more reliable way to obtain this information?
# How would the result change? Code it.

# histogram-based solution
for fish_id in fish_id_list:
    # filter dataframe using only data from current fish
    df_fish = df_all_fish[df_all_fish["fish_ID"] == fish_id]

    # compute histogram quantities
    hist_values, bin_edges = np.histogram(df_fish["x_position"])

    # center the bins
    bin_centers = center_bins_hist(bin_edges)

    # merge hist values in dark/undecided/light areas and for each area compute the width of its x-range
    i_dark_array = np.argwhere(bin_centers < dark_threshold)
    n_recordings_dark = np.sum(hist_values[i_dark_array])
    x_range_dark = np.abs(-4.5 - dark_threshold)

    i_undecided_array = np.argwhere(dark_threshold < bin_centers < light_threshold)
    n_recordings_undecided = np.sum(hist_values[i_undecided_array])
    x_range_undecided = np.abs(light_threshold - dark_threshold)

    i_light_array = np.argwhere(bin_centers > light_threshold)
    n_recordings_light = np.sum(hist_values[i_dark_array])
    x_range_light = np.abs(4.5 - light_threshold)

    # define list with results. Preference computed as relative number of recordings in each area
    result_list = [n_recordings_dark/x_range_dark,
                   n_recordings_undecided/x_range_undecided,
                   n_recordings_light/x_range_light]

    # find preference
    i_preference = np.argmax(result_list)

    # log preference
    if i_preference == 0:
        print(f"fish {fish_id} prefers DARKNESS")
    elif i_preference == 1:
        print(f"fish {fish_id} is UNDECIDED")
    elif i_preference == 2:
        print(f"fish {fish_id} prefers LIGHT")


# %% EXTRA: df.diff()
# Find how much time each fish spent in each area
for fish_id in fish_id_list:
    # filter dataframe using only data from current fish
    df_fish = df_all_fish[df_all_fish["fish_ID"] == fish_id]

    # compute time between recordings and assign it to new column
    delta_time = df_fish["time"].diff()
    df_fish["delta_time"] = delta_time

    # compute sum of time spent in each area
    time_in_dark = np.sum(df_fish[
                              df_fish["x_position"] < dark_threshold
                          ]["delta_time"])
    time_undecided = np.sum(df_fish[
                                np.logical_and(df_fish["x_position"] > dark_threshold,
                                               df_fish["x_position"] < light_threshold)
                            ]["delta_time"])
    time_in_light = np.sum(df_fish[
                               df_fish["x_position"] > light_threshold
                           ]["delta_time"])

    # log results
    print(f"fish {fish_id} spent {time_in_dark:.02f}s in darkness")
    print(f"fish {fish_id} spent {time_undecided:.02f}s close to the edge")
    print(f"fish {fish_id} spent {time_in_light:.02f}s in light")



# %% EXERCISE
# Check how undecided are the fish

# implement function to check when the fish is changing side
def compute_changing_side(df):
    # initialize array of zeros. We will update the values to 1 whenever the fish changes side
    changing_side_array = np.zeros(len(df))

    # loop over rows in the dataframe
    for i_row in range(1, len(df)):
        # check if side has changed between two consecutive recordings
        if df.iloc[i_row]["side"] * df.iloc[i_row-1]["side"] <= 0:
            # in case this is true, update the array with information on changing_side
            changing_side_array[i_row] = 1

    # add a column to the df containing the information about changing side
    df["changing_side"] = changing_side_array


# %% EXERCISE
# Quantify undecidedness of the fish

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
    print(f"fish {fish_id} changed side in {percentage_changing_side:.02f}% of the cases when presented the chance")


# %% EXTRA: groupby
# Use groupby to efficiently compute the mean and standard deviation of y_position to check that fish has no vertical
# preference

# compute mean of all columns grouping data by fish_ID
df_all_fish_mean = df_all_fish.groupby("fish_ID").mean()

# compute std of all columns grouping data by fish_ID
df_all_fish_std = df_all_fish.groupby("fish_ID").std()

# extract mean and std of y_position for each fish
mean_y_position = df_all_fish_mean["y_position"]
std_y_position = df_all_fish_std["y_position"]

# log result for each fish
for i_fish in range(len(mean_y_position)):
    print(f"fish {i_fish} | y_position | mean: {mean_y_position[i_fish]:.02f} | std: {std_y_position[i_fish]:.02f}")

# PRO-TIP: sometimes mean and std are not enough to describe your data. They are only good metrics if your data has a
# gaussian distribution


