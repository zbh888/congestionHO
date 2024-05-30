import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import seaborn as sns
from matplotlib import colormaps
from adjustText import adjust_text
import scipy.stats as stats

LEGEND_SIZE = 8
COLOR = 'tab20'

def escape_underscores(setting):
    input_string = "-".join(setting)
    escaped_string = input_string.replace("_", "-")
    return escaped_string


def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate the confidence interval for the mean of the given data.

    Parameters:
    data (list or numpy array): List of numbers.
    confidence (float): Confidence level, default is 0.95.

    Returns:
    tuple: (mean, lower bound of confidence interval, upper bound of confidence interval)
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    margin_of_error = sem * stats.t.ppf((1 + confidence) / 2., n - 1)

    return mean, margin_of_error


def highest_25_percent_confidence_interval(nnumbers):
        # Sort the list in descending order
    sorted_numbers = sorted(nnumbers, reverse=True)

        # Calculate the index to split the top 25%
    cutoff_index = int(len(sorted_numbers) * 0.25)

        # Select the highest 25%
    top_25_percent = sorted_numbers[:cutoff_index]

        # Calculate the mean and variance
    mean, margin_of_error = calculate_confidence_interval(top_25_percent, 0.95)

    return mean, margin_of_error, top_25_percent[-1]

def Non_empty_confidence_interval(nnumbers):

    # Calculate the mean and variance
    mean, margin_of_error = calculate_confidence_interval(nnumbers, 0.95)

    return mean, margin_of_error


def generate_numerical_results(results, filter_flag, filter_threshold):
    file_path = "aggregated_result.txt"
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} has been regenerated.")
    data = []
    for setting in results:
        source_alg = setting[0]
        candidate_alg = setting[1]
        ue_alg = setting[2]
        res = results[setting]
        time_sat_matrix = res['time_sat_matrix']
        time_sat_matrix_flatten = time_sat_matrix.flatten()
        x = time_sat_matrix_flatten[time_sat_matrix_flatten != 0]
        mean, margin_of_error, cutoff = highest_25_percent_confidence_interval(x)
        results[setting]['cutoff'] = cutoff
        with open("aggregated_result.txt", "a") as file:
            file.write(f"===== {source_alg} {candidate_alg} {ue_alg} =====\n")
            file.write(f"Maximum signalling: {np.max(time_sat_matrix)}\n")
            file.write(f"Total signalling: {np.sum(time_sat_matrix)}\n")
            file.write(f"Total handover: {res['total_handover']}\n")
            file.write(f"Non-Empty time: {np.sum(time_sat_matrix_flatten != 0)}\n")
            file.write(f"Non-Empty time top 25% confidence: {mean} ± {margin_of_error}\n")
        data.append((np.max(time_sat_matrix), setting))
    if filter_flag:
        new_result = {}
        sorted_setting = sorted(data, key=lambda x: x[0])
        print("Those with small max signalling")
        for element in sorted_setting[:int(filter_threshold*len(sorted_setting))]:
            setting = element[1]
            print(setting)
            new_result[setting] = results[setting]
        print("Those with large max signalling")
        for element in sorted_setting[-int(filter_threshold*len(sorted_setting)):]:
            setting = element[1]
            print(setting)
            new_result[setting] = results[setting]
        return new_result
    else:
        return results



def draw_total_load_each_satellite(results):
    print("Are there certain satellites handling much more signalling than others?")
    plt.figure(figsize=(10, 6))
    cmap = colormaps.get_cmap(COLOR)
    colors = [cmap(i) for i in range(len(results))]
    for idx, setting in enumerate(results):
        legend = escape_underscores(setting)
        res = results[setting]
        time_sat_matrix = res['time_sat_matrix']
        x = np.sum(time_sat_matrix, axis=1)
        sorted_data = np.sort(x)
        plt.plot(sorted_data, marker='', linestyle='-', color=colors[idx], linewidth=1, label=legend)
    plt.xlabel('Index')
    plt.ylabel('Total signalling load')
    plt.title('Sorted total signalling load each satellite')
    plt.grid(True)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.show()

def draw_cumulative_load_each_time(results):
    print("Given a signalling load, is the method showing that majority of the time slots are under the expected threshold?")
    plt.figure(figsize=(10, 6))
    cmap = colormaps.get_cmap(COLOR)
    colors = [cmap(i) for i in range(len(results))]
    for idx, setting in enumerate(results):
        legend = escape_underscores(setting)
        res = results[setting]
        time_sat_matrix = res['time_sat_matrix']
        time_sat_matrix_flatton = time_sat_matrix.flatten()
        x = time_sat_matrix_flatton[time_sat_matrix_flatton != 0]
        num_bins = 50
        counts, bin_edges = np.histogram(x, bins=num_bins, density=True)
        cdf = np.cumsum(counts * np.diff(bin_edges))
        plt.plot(bin_edges[1:], cdf, marker='none', linestyle='-', color=colors[idx], linewidth=1, label = legend)
    plt.xlabel('Signalling load each slot')
    plt.ylabel('Probability')
    plt.title('Cumulative plot for signalling load each slot')
    plt.grid(True)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.show()

def draw_busy_hour_distribution(results):
    print("Are there certain satellites handling majority of the busy (top 25%) signalling slots? ")
    plt.figure(figsize=(10, 6))
    cmap = colormaps.get_cmap(COLOR)
    colors = [cmap(i) for i in range(len(results))]
    for idx, setting in enumerate(results):
        legend = escape_underscores(setting)
        res = results[setting]
        time_sat_matrix = res['time_sat_matrix']
        mask = time_sat_matrix >= res['cutoff']
        count_greater_than_cutoff = np.sum(mask, axis=1)
        count_greater_than_cutoff = count_greater_than_cutoff / np.sum(count_greater_than_cutoff)
        sorted_data = np.sort(count_greater_than_cutoff)
        plt.plot(sorted_data, marker='', linestyle='-', linewidth=1, color=colors[idx], label=legend)
    plt.xlabel('Index')
    plt.ylabel('Busy slot count / Total number of busy slots')
    plt.title('Sorted busy slot share percentage each satellite')
    plt.grid(True)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.show()

def draw_heatmap(results, interval):
    intermediate_res = {}
    maximum = 0
    minimum = sys.maxsize
    for setting in results:
        legend = escape_underscores(setting)
        res = results[setting]
        time_sat_matrix = res['time_sat_matrix']
        N_SAT = time_sat_matrix.shape[0]
        N_TIME = time_sat_matrix.shape[1]
        assert (N_TIME % interval == 0)
        total_slots = N_TIME // interval
        res_reshaped = time_sat_matrix.reshape(N_SAT, total_slots, interval)
        res_reshaped_result = np.sum(res_reshaped, axis=2)
        intermediate_res[legend] = res_reshaped_result
        maximum = max(np.max(res_reshaped_result), maximum)
        minimum = min(np.min(res_reshaped_result), minimum)
    fig, axes = plt.subplots(ncols=len(intermediate_res), figsize=(3 * len(intermediate_res), 3))
    if len(intermediate_res) == 1:
        axes = [axes]  # Ensure axes is a list if only one subplot
    for ax, (legend, data) in zip(axes, intermediate_res.items()):
        sns.heatmap(data, ax=ax, vmin=minimum, vmax=maximum, cbar=False, cmap="YlGnBu", xticklabels=False, yticklabels=False)
        ax.set_title(legend)
    plt.tight_layout()
    plt.show()

def draw_max_access_slot(results):
    plt.figure(figsize=(10, 6))
    cmap = colormaps.get_cmap(COLOR)
    colors = [cmap(i) for i in range(len(results))]
    for idx, setting in enumerate(results):
        legend = escape_underscores(setting)
        res = results[setting]
        max_access_list = res['max_delays']
        sorted_data = np.sort(max_access_list)
        plt.plot(sorted_data, marker='', linestyle='-', linewidth=1, color=colors[idx], label=legend)
    plt.xlabel('Index')
    plt.ylabel('Time slot')
    plt.title('Max_delay')
    plt.grid(True)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.show()

def draw_max_signalling(results):
    print("Are there certain satellites experiencing higher signalling peaks than others?")
    plt.figure(figsize=(10, 6))
    cmap = colormaps.get_cmap(COLOR)
    colors = [cmap(i) for i in range(len(results))]
    for idx, setting in enumerate(results):
        legend = escape_underscores(setting)
        res = results[setting]
        time_sat_matrix = res['time_sat_matrix']
        x = np.max(time_sat_matrix, axis=1)
        sorted_data = np.sort(x)
        plt.plot(sorted_data, marker='', linestyle='-', linewidth=1, color=colors[idx], label=legend)
    plt.xlabel('Index')
    plt.ylabel('Signalling count')
    plt.title('Max signalling each satellite')
    plt.grid(True)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.show()

def draw_max_reservation(results):
    print("Are certain satellites experienced a much higher maximum reservation rate than others?")
    plt.figure(figsize=(10, 6))
    cmap = colormaps.get_cmap(COLOR)
    colors = [cmap(i) for i in range(len(results))]
    for idx, setting in enumerate(results):
        legend = escape_underscores(setting)
        res = results[setting]
        max_access_list = res['max_reservation']
        sorted_data = np.sort(max_access_list)
        plt.plot(sorted_data, marker='', linestyle='-', linewidth=1, color=colors[idx], label=legend)
    plt.xlabel('Index')
    plt.ylabel('Reservation rate')
    plt.title('Reservation rate')
    plt.grid(True)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.show()

def draw_numerical_result(results):
    text_size = 6
    cmap = colormaps.get_cmap(COLOR)
    colors = [cmap(i) for i in range(len(results))]
    data = []
    for idx, setting in enumerate(results):
        legend = escape_underscores(setting)
        res = results[setting]
        time_sat_matrix = res['time_sat_matrix']
        maximum_signalling = np.max(time_sat_matrix)
        total_signalling = np.sum(time_sat_matrix)
        time_sat_matrix_flatten = time_sat_matrix.flatten()
        x = time_sat_matrix_flatten[time_sat_matrix_flatten != 0]
        #mean, margin = Non_empty_confidence_interval(x)
        mean, margin, cutoff = highest_25_percent_confidence_interval(x)
        data.append((maximum_signalling, legend, colors[idx], total_signalling, mean, margin))
    sorted_data_triples = sorted(data, key=lambda x: x[0])
    sorted_data, sorted_labels, sorted_colors, sorted_totalsignalling, sorted_mean, sorted_margin = zip(*sorted_data_triples)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_data, marker='o', linestyle='-')
    texts = []
    for i, (label, color) in enumerate(zip(sorted_labels, sorted_colors)):
        texts.append(plt.text(i, sorted_data[i], label, fontsize=text_size, fontweight='bold', ha='right', va='bottom', color=color))
    adjust_text(texts)
    plt.title('Sorted maximum signalling')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()


    # Use sorted_data_triples to plot other side effects following the same order
    # The purpose is to learn the trade-off
    print("Overall, are we generating more signalling?")
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_totalsignalling, marker='o', linestyle='-')
    texts = []
    for i, (label, color) in enumerate(zip(sorted_labels, sorted_colors)):
        texts.append(plt.text(i, sorted_totalsignalling[i], label, fontsize=text_size, fontweight='bold', ha='right', va='bottom', color=color))
    adjust_text(texts)
    plt.title('Total signalling following main objective order')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

    print("During busy time (top 25%), are we making the busy time more busy?")
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_mean, marker='o', linestyle='-')
    plt.errorbar(range(len(sorted_mean)), sorted_mean, yerr=sorted_margin, fmt='o', ecolor='r', capsize=5)
    texts = []
    for i, (label, color) in enumerate(zip(sorted_labels, sorted_colors)):
        texts.append(plt.text(i, sorted_mean[i], label, fontsize=text_size, fontweight='bold', ha='right', va='bottom', color=color))
    adjust_text(texts)
    plt.xlabel('Index')
    plt.ylabel('Mean Value')
    plt.title('Mean with confidence Intervals')
    plt.grid(True)

