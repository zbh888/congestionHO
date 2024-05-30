import matplotlib.pyplot as plt
from Config import *
import pickle


class allCounters:
    def __init__(self, satellites, UEs):
        self.result = {}
        self.all_counter = {}
        self.UEs = UEs
        self.satellites = satellites
        for satid in satellites:
            self.all_counter[satid] = satellites[satid].counter
        self.N_SAT = len(satellites)
        self.N_TIME = satellites[0].DURATION
        self.generate_time_sat_matrix()
        self.generate_total_handover()
        self.generate_max_delay()
        self.generate_max_reservation_rate()

    def generate_max_reservation_rate(self):
        max_reservation = []
        for satid in self.satellites:
            max_reservation.append(self.satellites[satid].max_reservation_rate)
        self.result['max_reservation'] = max_reservation

    def generate_max_delay(self):
        max_delay = []
        for satid in self.satellites:
            max_delay.append(self.satellites[satid].record_max_delay)
        self.result['max_delays'] = max_delay


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
        self.result['time_sat_matrix'] = res

    def generate_total_handover(self):
        total_handover_count = 0
        for ueid in self.UEs:
            ue = self.UEs[ueid]
            total_handover_count += (len(ue.serving_satellite_history) - 1)
        self.result['total_handover'] = total_handover_count

    def generate_delay_box(self):
        total = []
        for ueid in self.UEs:
            ue = self.UEs[ueid]
            total.append(sum(ue.applied_delay_history) / len(ue.applied_delay_history))
        self.result['delay_box'] = total
        plt.boxplot(total)
        plt.ylabel('Delay')
        plt.savefig('box.png')
        plt.close()

    def give_result(self):
        with open(RESULT_PATH, 'wb') as file:
            pickle.dump(self.result, file)


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
