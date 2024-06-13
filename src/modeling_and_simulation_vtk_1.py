import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#%%
def simulate_animal_movement(timestamps=1000, arena_size=1, start_x=0.5, start_y=0.5, threshold_right=0.5, threshold_up=0.5):

    # Initialize lists to store the positions and swim numbers
    swim_number = [0]
    animal_position_x = [start_x]
    animal_position_y = [start_y]

    # Simulate the animal movement
    for t in np.arange(timestamps):
        x_bias = np.random.uniform(0, 1)
        y_bias = np.random.uniform(0, 1)

        dx = np.random.normal(0.05, 0.01)
        dy = np.random.normal(0.05, 0.01)

        prev_position_x = animal_position_x[-1]
        prev_position_y = animal_position_y[-1]

        if x_bias > threshold_right:
            new_position_x = prev_position_x + dx
        else:
            new_position_x = prev_position_x - dx

        if y_bias > threshold_up:
            new_position_y = prev_position_y + dy
        else:
            new_position_y = prev_position_y - dy

        # Ensure the animal stays within the arena bounds
        new_position_x = max(0, min(arena_size, new_position_x))
        new_position_y = max(0, min(arena_size, new_position_y))

        animal_position_x.append(new_position_x)
        animal_position_y.append(new_position_y)
        swim_number.append(t + 1)

    # Now we put everything in a dataframe
    model_df = pd.DataFrame()

    model_df['swim_number'] = swim_number
    model_df['position_x'] = animal_position_x
    model_df['position_y'] = animal_position_y

    return model_df
#%%


model_df = simulate_animal_movement(timestamps=1000)

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
    1. Make the animal move towards the right more than the left.
    2. Make the animal move more towards the bottom left.
    
    Verify results using plots.
"""