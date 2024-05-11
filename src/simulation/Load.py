import json
import struct

import numpy as np


class UE_template:
    def __init__(self, ID, x, y):
        self.ID = ID
        self.x = x
        self.y = y


class Satellite_template:
    def __init__(self, ID, x, y, h, v, r, sind, cosd):
        self.ID = ID
        self.x = x
        self.y = y
        self.h = h
        self.v = v
        self.r = r
        self.sind = sind
        self.cosd = cosd


def load_scenario(feasible):
    UEs = []
    satellites = []

    # Read JSON data from file
    with open('../generateScenario/satellites.json', 'r') as fileSAT:
        satellites_data = json.load(fileSAT)

    # Process JSON data
    for sat_data in satellites_data:
        # Access satellite parameters
        s = Satellite_template(
            ID=sat_data['index'],
            x=sat_data['x'],
            y=sat_data['y'],
            h=sat_data['h'],
            v=sat_data['v'],
            r=sat_data['r'],
            sind=sat_data['sind'],
            cosd=sat_data['cosd'])
        satellites.append(s)

    with open('../generateScenario/UEs.json', 'r') as fileUE:
        UEs_data = json.load(fileUE)

    # Process JSON data
    for UE_data in UEs_data:
        # Access satellite parameters
        ue = UE_template(
            ID=UE_data['index'],
            x=UE_data['x'],
            y=UE_data['y'])
        UEs.append(ue)

    C = np.load('../generateScenario/simulation_coverage_info.npy')

    return UEs, satellites, C
