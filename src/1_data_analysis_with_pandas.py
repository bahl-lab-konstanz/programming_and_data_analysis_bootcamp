# DATA ANALYSIS WITH PANDAS

# For this section we will need an external library, pandas. So, we will use an "import" statement, that provides
# our script with the requested package from our conda environment. For convenience, we will use an abbreviation using
# the "as" keyword to give the imported library an alias
import pandas as pd

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
path = "./data/fish_100.csv"

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
    df = pd.read_csv(path, index_col=0, header=0)
    if show_preview:
        print(df.head())
    return df


# %% And test the function
df = import_csv(path, show_preview=True)


# %% pandas provides an easy way to access basic statistical characteristics of our data. In particular dataframes have
# built-in methods "min()", "max()", "mean()", "sum()", "corr()", and "describe()" which gives a summary.
# Print basic statistics of the columns "average_speed" and "start_time"
print("AVERAGE SPEED")
print(df["average_speed"].describe())

print("START TIME")
print(df["start_time"].describe())


# %% The dataset we just used come from a single fish. For our analysis we need to put together data from different
# individuals
def import_csv_from_dir(dir_path, show_info=False):
    data_list = []
    for fish_dataset_path in dir_path.glob("*.csv"):
        # load data in a temporary structure
        df_temp = import_csv(fish_dataset_path)
        data_list.append(df_temp)
    # merge all the dataframes in data_list and print information about the dataframe
    df = pd.concat(data_list, axis=0)
    if show_info:
        print(df.info())
    return df


# %% Test the function
dir_path = Path("./data")
df_all_fish = import_csv_from_dir(dir_path, show_info=True)


# %% Implement a function to add a column to the dataframe, telling us in which side the fish is at the end of the swim
def where_is_the_fish():
    pass


# %% Implement a function to add a column to the dataframe, telling us whether the fish just changed side
def is_the_fish_changing_side():
    pass



# ALL WHAT COMES NEXT IS DEPRECATED BUT I COULD KEEP THE STRUCTURE

# %% Now that we have a proper dataframe, we can extract some information. For example, we can now check the accuracy of
# each fish using the amount of 1s in correct_bout. We are going to simplify the analysis as much as possible,
# so we will only consider left or right swims, filtering out all the straight ones, identified in the dataframe by the
# value -1 in column "correct_bout"
df_all_fish = df_all_fish[~(df_all_fish["correct_bout"] == -1)]


# %% we iterate over fishes
for fish_id in df_all_fish.index.unique("fish_ID"):
    # we select only the subset of rows corresponding to the present fish, using all the trials
    df_fish = df_all_fish.xs(fish_id, level="fish_ID")
    # and now we compute the accuracy as correct_bouts/total_bouts
    accuracy = df_fish["correct_bout"].sum() / len(df_fish["correct_bout"])
    print(f"INFO | fish {fish_id} has accuracy = {accuracy}")

# %% we can also perform the same operation using the groupby method
df_accuracy = df.groupby("fish_ID")["correct_bout"].mean()

# %% from the output dataframe we can also obtain information on the best and worst performing fish
print(f"INFO | fish {df_accuracy.idxmax()} has accuracy {df_accuracy.max()}. It's the best")
print(f"INFO | fish {df_accuracy.idxmin()} has accuracy {df_accuracy.min()}. It's the worst")


# %% Finally we are interested in looking at how the best and worst fish move in the arena, so we are going to plot the
# swimming trajectory for these two

# import list containing color names
from src.utils.constants import color_list

#  define the index of the best and worst fish
fish_id_best = "insert here your answer"
fish_id_worst = "insert here your answer"

# define grid for plotting
fig, axs = plt.subplots(1, 2)

axs[0].set_title("best fish trajectory")
axs[0].set_ylabel("start_y_position")
axs[0].set_xlim([-1, 1])
axs[0].set_ylim([-1, 1])
# plot the trajectories for the best fish
for trial in df.index.unique("trial"):
    df.xs((fish_id_best, trial), level=["fish_ID", "trial"]).plot(x="start_x_position", y="start_y_position",
                                                                  ax=axs[0], kind="line", legend=False,
                                                                  color=color_list[trial])

axs[1].set_title("worst fish trajectory")
axs[1].set_xlim([-1, 1])
axs[1].set_ylim([-1, 1])
# plot the trajectory for the worst fish
for trial in df.index.unique("trial"):
    df.xs((fish_id_worst, trial), level=["fish_ID", "trial"]).plot(x="start_x_position", y="start_y_position",
                                                                   ax=axs[1], kind="line", legend=False,
                                                                   color=color_list[trial])

# show plots
plt.show()




def create_target_column():
    pass

def preprocess(csv_path):
    df = import_csv(csv_path)
    create_target_column(df)
    return df