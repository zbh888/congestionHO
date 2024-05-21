import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class allCounters:
    def __init__(self, satellites, UEs):
        self.all_counter = {}
        self.UEs = UEs
        for satid in satellites:
            self.all_counter[satid] = satellites[satid].counter
        self.N_SAT = len(satellites)
        self.N_TIME = satellites[0].DURATION
        self.time_sat_matrix = self.generate_time_sat_matrix()
        self.time_sat_matrix_flatten = self.time_sat_matrix.flatten()

    def generate_time_sat_matrix(self):
        res = []
        for satid in range(self.N_SAT):
            sat_res = []
            for t in range(self.N_TIME):
                count = 0
                counter_sat_t = self.all_counter[satid].counter[t]
                for header in counter_sat_t:
                    count += counter_sat_t[header]
                sat_res.append(count)
            res.append(sat_res)
        return np.array(res)

    def generate_heap_map(self, interval):
        assert (self.N_TIME % interval == 0)
        total_slots = self.N_TIME // interval
        res_reshaped = self.time_sat_matrix.reshape(self.N_SAT, total_slots, interval)
        result = np.sum(res_reshaped, axis=2)

        plt.figure(figsize=(100, 80))
        sns.heatmap(result, annot=False, cmap='coolwarm')
        plt.savefig('heatmap.png')
        plt.close()

    def generate_cumulative_load_each_time(self):
        x = self.time_sat_matrix_flatten[self.time_sat_matrix_flatten != 0]
        num_bins = 100
        counts, bin_edges = np.histogram(x, bins=num_bins, density=True)
        cdf = np.cumsum(counts * np.diff(bin_edges))
        plt.plot(bin_edges[1:], cdf, marker='none', linestyle='-')
        plt.grid(True)
        plt.savefig('cumulative.png')
        plt.close()


    def generate_delay_box(self):
        total = []
        for ueid in self.UEs:
            ue = self.UEs[ueid]
            total.append(sum(ue.applied_delay_history) / len(ue.applied_delay_history))
        plt.boxplot(total)
        plt.ylabel('Delay')
        plt.savefig('box.png')
        plt.close()

    def generate_total_load_each_satellite(self):
        x = np.sum(self.time_sat_matrix, axis = 1)
        sorted_data = np.sort(x)
        plt.plot(sorted_data, marker='', linestyle='-', color='b')
        plt.grid(True)
        plt.savefig('total_each_satellite.png')
        plt.close()


    def generate_total_handover(self):
        total_handover_count = 0
        for ueid in self.UEs:
            ue = self.UEs[ueid]
            total_handover_count += (len(ue.serving_satellite_history) - 1)
        return total_handover_count

    def generate_total_signalling(self):
        return np.sum(self.time_sat_matrix)

    def highest_25_percent_mean_variance(self, nnumbers):
        # Sort the list in descending order
        sorted_numbers = sorted(nnumbers, reverse=True)

        # Calculate the index to split the top 25%
        cutoff_index = int(len(sorted_numbers) * 0.25)

        # Select the highest 25%
        top_25_percent = sorted_numbers[:cutoff_index]

        # Calculate the mean and variance
        mean_top_25 = np.mean(top_25_percent)
        variance_top_25 = np.var(top_25_percent)

        return mean_top_25, variance_top_25

    def give_result(self, interval):
        x = self.time_sat_matrix_flatten[self.time_sat_matrix_flatten != 0]
        mean, var = self.highest_25_percent_mean_variance(x)
        with open("result_stat.txt", "w") as file:
            file.write(f"Total signalling: {self.generate_total_signalling()}\n")
            file.write(f"Total handover: {self.generate_total_handover()}\n")
            file.write(f"Non-Empty time: {np.sum(self.time_sat_matrix_flatten != 0)}\n")
            file.write(f"Non-Empty time top 25% mean: {mean}\n")
            file.write(f"Non-Empty time top 25% variance: {var}\n")
        self.generate_heap_map(interval)
        self.generate_delay_box()
        self.generate_cumulative_load_each_time()
        self.generate_total_load_each_satellite()
        # there should be a measurement to see if busy time all gathering to one satellite, a list containing [10, 20, 0,0] each enty is satellite




class counter:
    def __init__(self, duration):
        self.counter = {}
        for i in range(duration):
            self.counter[i] = {}

    def increment(self, header, time):
        if header not in self.counter[time]:
            self.counter[time][header] = 1
        else:
            self.counter[time][header] += 1
