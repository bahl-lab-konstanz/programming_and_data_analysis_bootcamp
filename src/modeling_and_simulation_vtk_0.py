import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

# matplotlib.use('macOSX')

# %%

# Setting up the arena and the time

timestamps = 1000  # Number of timestamps for which the stimulus is run

arena_size = 1  # Arena is a 1 x 1 unit square

# %%

# Setting up the animal
start_x = 0.5  # Starting position of the animal in x-axis
start_y = 0.5  # Starting position of the animal in x-axis

swim_number = [0]  # A list to store the swim number of the animal
animal_position_x = [start_x]  # A list to store the animal position in x-axis
animal_position_y = [start_y]  # A list to store the animal position in y-axis

threshold_right = 0.5  # Average left/right bias (x-axis)
threshold_up = 0.5  # Average up/down bias (y-axis)


# %%

# Simulate the animal movement
for t in np.arange(timestamps):

    x_bias = np.random.uniform(0, 1)  # for every move the animal has a bias which decides animal goes to left or right
    y_bias = np.random.uniform(0, 1)  # for every move the animal has a bias which decides animal goes to up or down

    prev_position_x = animal_position_x[-1]  # storing old position in x-axis
    prev_position_y = animal_position_y[-1]  # storing old position in y-axis

    dx = 0.05  # Speed of animal in x-axis
    dy = 0.05  # Speed of animal in y-axis

    # dx = np.random.normal(0.05, 0.01)
    # dy = np.random.normal(0.05, 0.01)

    if x_bias > threshold_right:
        new_position_x = prev_position_x + dx  # New position = Old position + step_right
    else:
        new_position_x = animal_position_x[-1] - dx  # New position = Old position + step_left

    if y_bias > threshold_up:
        new_position_y = animal_position_y[-1] + dy  # New position = Old position + step_up
    else:
        new_position_y = animal_position_y[-1] - dy  # New position = Old position + step

    # Make sure the animal is within the arena
    new_position_x = max(0, min(arena_size, new_position_x))
    new_position_y = max(0, min(arena_size, new_position_y))

    animal_position_x.append(new_position_x)  # Now we put the new position into the storage list
    animal_position_y.append(new_position_y)  # Now we put the new position into the storage list
    swim_number.append(t+1)

# Now we put everything in a dataframe
model_df = pd.DataFrame()

model_df['swim_number'] = swim_number
model_df['position_x'] = animal_position_x
model_df['position_y'] = animal_position_y
#%%

# First figure
plt.figure()  # Create a new figure
plt.plot(model_df['position_x'], model_df['position_y'], alpha=0.8)
plt.scatter(model_df['position_x'], model_df['position_y'], alpha=0.8, s=2)
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.title('Animal trajectory')
plt.xlabel('Position X')
plt.ylabel('Position Y')
plt.show()

# Second figure
plt.figure()  # Create another new figure
plt.hist(model_df['position_x'])  # Plot
plt.title('Distribution of  position x')
plt.xlabel('Position X')
plt.ylabel('Occurrences')
plt.show()

#%%

"""
    Make the animal move faster in the y-axis and slower in the x-axis.
"""

""" 
    Here we can see that the model animal is moving always at 45 degrees.
    It takes one step up or down and one step left or right. This quite unrealistic. 
    Animals show variable motion - sometimes moving fast, sometimes stopping, etc.
    How can we incorporate this into our model?
                                                                                        """
