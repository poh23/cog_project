import math
import csv
import matplotlib.pyplot as plt
import numpy as np
import string
alphabet = list(string.ascii_uppercase)
from num2words import num2words

def calculate_distance(file_path):
    x_pos = []
    y_pos = []
    total_distance = 0.0

    with open(file_path, 'r') as file:
        reader = csv.DictReader(file, fieldnames=['t', 'x', 'y', 'r'])

        # Initialize previous coordinates
        prev_x, prev_y = None, None

        for row in reader:
            x = float(row['x'])
            y = float(row['y'])
            x_pos.append(x)
            y_pos.append(y)

            if prev_x is not None and prev_y is not None:
                # Calculate the distance between the current point and the previous one
                distance = math.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                total_distance += distance

            # Update previous coordinates
            prev_x, prev_y = x, y

    return total_distance, x_pos, y_pos


def calculate_avg_distance_several_ants(num):
    file_names = alphabet[:num]
    avg_distances = []
    for name in file_names:
        file_path = f'data/{num2words(num)}_ants_ant_{name}.txt'
        dist, x, y = calculate_distance(file_path)
        avg_distances.append(dist)

    return np.mean(avg_distances), np.std(avg_distances), np.max(avg_distances), np.min(avg_distances)


def plot_farthest_and_lowest_distances(num_ants, farthest_distances, lowest_distances):
    """
    Plots a bar chart of the farthest and lowest distances walked by ants for different numbers of ants.

    Parameters:
    - num_ants: List or array of the number of ants.
    - farthest_distances: List or array of the farthest distances corresponding to the number of ants.
    - lowest_distances: List or array of the lowest distances corresponding to the number of ants.
    """
    # Convert input data to numpy arrays
    num_ants = np.array(num_ants)
    farthest_distances = np.array(farthest_distances)
    lowest_distances = np.array(lowest_distances)

    # Define bar width and positions for side-by-side bars
    bar_width = 0.3
    index = np.arange(len(num_ants))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(index, farthest_distances, width=bar_width, color='skyblue', edgecolor='black', label='Farthest Distance')
    plt.bar(index + bar_width, lowest_distances, width=bar_width, color='lightcoral', edgecolor='black',
            label='Lowest Distance')

    # Customize plot appearance for scientific style
    plt.xlabel('Number of Ants', fontsize=14)
    plt.ylabel('Distance', fontsize=14)
    plt.title('Farthest and Lowest Distances for Different Numbers of Ants', fontsize=16)
    plt.xticks(index + bar_width / 2, num_ants, fontsize=12)  # Center the x-ticks between the groups
    plt.yticks(fontsize=12)

    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add legend
    plt.legend(fontsize=12)

    # Set tight layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_bar_chart_with_error_bars(num_ants, avg_distances, std_devs):
    """
    Plots a bar chart of average distance with standard deviation error bars for different numbers of ants.

    Parameters:
    - num_ants: List or array of the number of ants.
    - avg_distances: List or array of average distances corresponding to the number of ants.
    - std_devs: List or array of standard deviations of the average distances.
    """
    # Convert input data to numpy arrays
    num_ants = np.array(num_ants)
    avg_distances = np.array(avg_distances)
    std_devs = np.array(std_devs)

    # Define bar width
    bar_width = 0.6

    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(num_ants, avg_distances, width=bar_width, color='skyblue', edgecolor='black', yerr=std_devs,
                   capsize=5, error_kw=dict(ecolor='black', lw=1.5))

    # Customize plot appearance for scientific style
    plt.xlabel('Number of Ants', fontsize=14)
    plt.ylabel('Average Distance', fontsize=14)
    plt.title('Average Distance with Standard Deviation for Different Numbers of Ants', fontsize=16)
    plt.xticks(num_ants, fontsize=12)
    plt.yticks(fontsize=12)

    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Set tight layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()


def generate_list_of_data():
    num_of_ants = []
    avg_distances = []
    dist_stds = []
    farthest_dists= []
    shortest_dists = []
    for i in range(1,8,1):
        num_of_ants.append(i)
        avg, std, far_dist, shrt_dist = calculate_avg_distance_several_ants(i)
        avg_distances.append(avg)
        dist_stds.append(std)
        farthest_dists.append(far_dist)
        shortest_dists.append(shrt_dist)
        print(f'{i} ant: average: {avg} mm, standard deviation: {std} mm, max distance by ant: {far_dist}, min distance by ant: {shrt_dist}')

    return num_of_ants, avg_distances, dist_stds, farthest_dists, shortest_dists

num_of_ants, avg_distances, dist_stds , farthest_dists, shortest_dists= generate_list_of_data()
plot_bar_chart_with_error_bars(num_of_ants, avg_distances,dist_stds)
plot_farthest_and_lowest_distances(num_of_ants, farthest_dists, shortest_dists)

