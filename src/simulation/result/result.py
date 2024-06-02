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


def coefficient_of_variation(data):
    if len(data) == 0:
        raise ValueError("The data list is empty.")

    mean = np.mean(data)
    std_dev = np.std(data)

    if mean == 0:
        raise ValueError("The mean of the data is zero, cannot compute coefficient of variation.")

    cv = std_dev / mean
    return cv


def calculate_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    margin_of_error = sem * stats.t.ppf((1 + confidence) / 2., n - 1)

    return mean, margin_of_error

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

def main_objective_compute_max_signalling(result):
    time_sat_matrix = result['time_sat_matrix']
    maximum_signalling = np.max(time_sat_matrix)
    return maximum_signalling

def side_effect_compute_total_siganlling(result):
    time_sat_matrix = result['time_sat_matrix']
    total_signalling = np.sum(time_sat_matrix)
    return total_signalling

def side_effect_compute_busy_time_balance_cv(result, cutoff_percent):
    time_sat_matrix = result['time_sat_matrix']
    time_sat_matrix_flatten = time_sat_matrix.flatten()
    x = time_sat_matrix_flatten[time_sat_matrix_flatten != 0]
    sorted_numbers = sorted(x, reverse=True)
    cutoff_index = int(len(sorted_numbers) * cutoff_percent)
    cutoff = sorted_numbers[:cutoff_index][-1]
    mask = time_sat_matrix >= cutoff
    count_greater_than_cutoff = np.sum(mask, axis=1)
    count_greater_than_cutoff_percent = count_greater_than_cutoff / np.sum(count_greater_than_cutoff)
    sorted_data = np.sort(count_greater_than_cutoff_percent)[len(count_greater_than_cutoff_percent)//2:]
    cv = coefficient_of_variation(sorted_data)
    return cv

def side_effect_compute_busy_time_confidence(result, cutoff_percent):
    time_sat_matrix = result['time_sat_matrix']
    time_sat_matrix_flatten = time_sat_matrix.flatten()
    x = time_sat_matrix_flatten[time_sat_matrix_flatten != 0]
    sorted_numbers = sorted(x, reverse=True)
    cutoff_index = int(len(sorted_numbers) * cutoff_percent)
    busy_time_signalling_count = sorted_numbers[:cutoff_index]
    mean, margin_of_error = calculate_confidence_interval(busy_time_signalling_count, 0.95)
    return mean, margin_of_error

def side_effect_compute_total_reservation(result):
    reservation_list = result['reservation_count']
    return np.sum(reservation_list)

def side_effect_compute_reservation_cv(result):
    reservation_list = result['reservation_count']
    sorted_data = np.sort(reservation_list)[len(reservation_list) // 2:]
    cv = coefficient_of_variation(sorted_data)
    return cv

def side_effect_compute_UE_access_mean(result):
    total_access = []
    UE_access_list = result['ue_delay_history']
    for list in UE_access_list:
        total_access += list
    return np.mean(total_access)

def side_effect_compute_UE_access_cv(result):
    total_access = []
    UE_access_list = result['ue_delay_history']
    for list in UE_access_list:
        total_access += list
    cv = coefficient_of_variation(total_access)
    return cv

def prepare_result(results, filter_flag, filter_threshold):
    busy_percent = 0.2
    file_path = "aggregated_result.txt"
    with open(file_path, "w") as file:
        file.write("S_ALG,C_ALG,UE_ALG,ACC_OPPORTUNITIES,MAX_SIGNALLING,TOTAL_SIGNALLING,BUSY_CV,BUSY_SIG_MEAN,BUSY_SIG_MARGIN,TOTAL_RESERVE,RESERVE_CV,DELAY_MEAN,DELAY_CV\n")
    data = []
    for setting in results:
        res = results[setting]
        maximum_signalling = main_objective_compute_max_signalling(res)
        total_signalling = side_effect_compute_total_siganlling(res)
        busy_time_balance_cv = side_effect_compute_busy_time_balance_cv(res, busy_percent)
        mean, margin = side_effect_compute_busy_time_confidence(res, busy_percent)
        total_reservation = side_effect_compute_total_reservation(res)
        reservation_balance_cv = side_effect_compute_reservation_cv(res)
        delay_mean = side_effect_compute_UE_access_mean(res)
        delay_cv = side_effect_compute_UE_access_cv(res)

        results[setting]['maximum_signalling'] = maximum_signalling
        data.append((maximum_signalling, setting))
        results[setting]['total_signalling'] = total_signalling
        results[setting]['busy_time_balance_cv'] = busy_time_balance_cv
        results[setting]['signalling_mean'] = mean
        results[setting]['signalling_margin'] = margin
        results[setting]['total_reservation'] = total_reservation
        results[setting]['reservation_balance_cv'] = reservation_balance_cv
        results[setting]['delay_mean'] = delay_mean
        results[setting]['delay_cv'] = delay_cv

        with open(file_path, "a") as file:
            file.write(f"{setting[0]},")
            file.write(f"{setting[1]},")
            file.write(f"{setting[2]},")
            file.write(f"{setting[3]},")
            file.write(f"{maximum_signalling},")
            file.write(f"{total_signalling},")
            file.write(f"{busy_time_balance_cv},")
            file.write(f"{mean},")
            file.write(f"{margin},")
            file.write(f"{total_reservation},")
            file.write(f"{reservation_balance_cv},")
            file.write(f"{delay_mean},")
            file.write(f"{delay_cv}\n")
    if filter_flag:
        new_result = {}
        sorted_setting = sorted(data, key=lambda x: x[0])
        print("Those with small max signalling")
        for element in sorted_setting[:int(filter_threshold * len(sorted_setting))]:
            setting = element[1]
            print(setting)
            new_result[setting] = results[setting]
        print("Those with large max signalling")
        for element in sorted_setting[-int(filter_threshold * len(sorted_setting)):]:
            setting = element[1]
            print(setting)
            new_result[setting] = results[setting]
        return new_result
    else:
        return results

def draw_prepared_result(results):
    text_size = 6
    cmap = colormaps.get_cmap(COLOR)
    colors = [cmap(i) for i in range(len(results))]
    order_setting = []
    for setting in results:
        res = results[setting]
        order_setting.append((res['maximum_signalling'], setting))
    sorted_objective_setting = sorted(order_setting, key=lambda x: x[0])
    sorted_labels = []
    sorted_colors = []
    for index, element in enumerate(sorted_objective_setting):
        setting = element[1]
        sorted_labels.append(escape_underscores(setting))
        sorted_colors.append(colors[index])

    # Plotting the main objective
    sorted_data = []
    for element in sorted_objective_setting:
        res = results[element[1]]
        sorted_data.append(res['maximum_signalling'])
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_data, marker='o', linestyle='-')
    texts = []
    for i, (label, color) in enumerate(zip(sorted_labels, sorted_colors)):
        texts.append(plt.text(i, sorted_data[i], label, fontsize=text_size, fontweight='bold', ha='right', va='bottom', color=color))
    adjust_text(texts)
    plt.title('Maximum signalling')
    #plt.xlabel('Index')
    plt.ylabel('Signalling count')
    plt.grid(True)
    plt.show()

    # Plotting side effects
    print("The total signalling: the lower, the better")
    sorted_data = []
    for element in sorted_objective_setting:
        res = results[element[1]]
        sorted_data.append(res['total_signalling'])
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_data, marker='o', linestyle='-')
    texts = []
    for i, (label, color) in enumerate(zip(sorted_labels, sorted_colors)):
        texts.append(plt.text(i, sorted_data[i], label, fontsize=text_size, fontweight='bold', ha='right', va='bottom',
                              color=color))
    adjust_text(texts)
    plt.title('Total Signalling')
    plt.xlabel('Index')
    plt.ylabel('Signalling count')
    plt.grid(True)
    plt.show()

    print("busy time slot shares: the lower, the better")
    sorted_data = []
    for element in sorted_objective_setting:
        res = results[element[1]]
        sorted_data.append(res['busy_time_balance_cv'])
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_data, marker='o', linestyle='-')
    texts = []
    for i, (label, color) in enumerate(zip(sorted_labels, sorted_colors)):
        texts.append(plt.text(i, sorted_data[i], label, fontsize=text_size, fontweight='bold', ha='right', va='bottom',
                              color=color))
    adjust_text(texts)
    plt.title('Busy time slot shares balance evaluation')
    plt.xlabel('Index')
    plt.ylabel('Coefficient of variation')
    plt.grid(True)
    plt.show()

    #
    print("busy time slot signalling count confidence interval: the lower, the better")
    sorted_mean = []
    sorted_margin = []
    for element in sorted_objective_setting:
        res = results[element[1]]
        sorted_mean.append(res['signalling_mean'])
        sorted_margin.append(res['signalling_margin'])
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_mean, marker='o', linestyle='-')
    plt.errorbar(range(len(sorted_mean)), sorted_mean, yerr=sorted_margin, fmt='o', ecolor='r', capsize=5)
    texts = []
    for i, (label, color) in enumerate(zip(sorted_labels, sorted_colors)):
        texts.append(plt.text(i, sorted_mean[i], label, fontsize=text_size, fontweight='bold', ha='right', va='bottom', color=color))
    adjust_text(texts)
    plt.xlabel('Index')
    plt.ylabel('Mean Value')
    plt.title('busy slot mean with confidence Intervals')
    plt.grid(True)

    print("Total reservation time: the lower, the better")
    sorted_data = []
    for element in sorted_objective_setting:
        res = results[element[1]]
        sorted_data.append(res['total_reservation'])
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_data, marker='o', linestyle='-')
    texts = []
    for i, (label, color) in enumerate(zip(sorted_labels, sorted_colors)):
        texts.append(plt.text(i, sorted_data[i], label, fontsize=text_size, fontweight='bold', ha='right', va='bottom',
                              color=color))
    adjust_text(texts)
    plt.title('Total reservation time')
    plt.xlabel('Index')
    plt.ylabel('Time slot count')
    plt.grid(True)
    plt.show()

    print("Reservation time balance: the lower the better")
    sorted_data = []
    for element in sorted_objective_setting:
        res = results[element[1]]
        sorted_data.append(res['reservation_balance_cv'])
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_data, marker='o', linestyle='-')
    texts = []
    for i, (label, color) in enumerate(zip(sorted_labels, sorted_colors)):
        texts.append(plt.text(i, sorted_data[i], label, fontsize=text_size, fontweight='bold', ha='right', va='bottom',
                              color=color))
    adjust_text(texts)
    plt.title('Reservation balance across satellites evaluation')
    plt.xlabel('Index')
    plt.ylabel('Coefficient of variation')
    plt.grid(True)
    plt.show()

    print("UE average access time: the lower, the better")
    sorted_data = []
    for element in sorted_objective_setting:
        res = results[element[1]]
        sorted_data.append(res['delay_mean'])
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_data, marker='o', linestyle='-')
    texts = []
    for i, (label, color) in enumerate(zip(sorted_labels, sorted_colors)):
        texts.append(plt.text(i, sorted_data[i], label, fontsize=text_size, fontweight='bold', ha='right', va='bottom',
                              color=color))
    adjust_text(texts)
    plt.title('UE access time mean')
    plt.xlabel('Index')
    plt.ylabel('Time slot count')
    plt.grid(True)
    plt.show()

    print("UE access time balance: the lower, the better")
    sorted_data = []
    for element in sorted_objective_setting:
        res = results[element[1]]
        sorted_data.append(res['delay_cv'])
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_data, marker='o', linestyle='-')
    texts = []
    for i, (label, color) in enumerate(zip(sorted_labels, sorted_colors)):
        texts.append(plt.text(i, sorted_data[i], label, fontsize=text_size, fontweight='bold', ha='right', va='bottom',
                              color=color))
    adjust_text(texts)
    plt.title('UE access time balance evaluation')
    plt.xlabel('Index')
    plt.ylabel('Coefficient of variation')
    plt.grid(True)
    plt.show()