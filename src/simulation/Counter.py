import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class allCounters:
    def __init__(self, satellites):
        self.all_counter = {}
        for satid in satellites:
            self.all_counter[satid] = satellites[satid].counter
        self.N_SAT = len(satellites)
        self.N_TIME = satellites[0].DURATION

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
        res = self.generate_time_sat_matrix()
        total_slots = self.N_TIME // interval
        res_reshaped = res.reshape(self.N_SAT, total_slots, interval)
        result = np.sum(res_reshaped, axis=2)

        plt.figure(figsize=(100, 80))
        sns.heatmap(result, annot=False, cmap='coolwarm')
        plt.savefig('heatmap.png')


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
