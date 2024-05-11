import pickle
import numpy as np

class Oracle:
    def __init__(self):
        with open('../optimizer/bho.pkl', 'rb') as inp:
            self.d = pickle.load(inp)
        self.N_UE = len(self.d["C"])
        self.N_SAT = len(self.d["C"][0])
        self.N_TIME = self.d["N_TIME"]
        self.h = self.d['h']
        self.o2i = self.d["o2i"]
        self.satellite_sequence = {}
        for ueid in range(self.N_UE):
            self.satellite_sequence[ueid] = []
            servingsat_time_array = np.array(self.h[ueid])
            for t in range(servingsat_time_array.shape[1]):
                for satid in range(servingsat_time_array.shape[0]):
                    if self.h[ueid][satid][t] == 1:
                        if len(self.satellite_sequence[ueid]) == 0:
                            self.satellite_sequence[ueid].append(satid)
                        elif self.satellite_sequence[ueid][-1] != satid:
                            self.satellite_sequence[ueid].append(satid)

    def query_next_satellite(ueid, serving_satellite_id):
        array = self.satellite_sequence[ueid]
        for index, satid in enumerate(array):
            if serving_satellite_id == satid:
                if index == len(array) - 1:
                    return -1 # error code for at the last satellite
                else:
                    # TODO consider if the UE will be covered by this one or not
                    # This is possible, but we will see
                    return array[index + 1]
            # does not find ueid should be served by serving_satellite_id
            # maybe out of sync
            return -2 
        