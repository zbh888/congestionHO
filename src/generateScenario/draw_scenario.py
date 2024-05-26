import json
import struct
import matplotlib.pyplot as plt
import numpy as np


class UE_template:
    def __init__(self, ID, x, y):
        self.ID = ID
        self.x = x
        self.y = y


class Satellite_template:
    def __init__(self, ID, x, y, h, v, r, sind, cosd, tid):
        self.ID = ID
        self.x = x
        self.y = y
        self.h = h
        self.v = v
        self.r = r
        self.sind = sind
        self.cosd = cosd
        self.tid = tid

    def update_location(self, time):
        dis = self.v * time
        y_dis = dis * self.sind
        x_dis = dis * self.cosd
        self.x = self.x + x_dis
        self.y = self.y + y_dis


def load_scenario():
    UEs = []
    satellites = []

    # Read JSON data from file
    with open('satellites.json', 'r') as fileSAT:
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
            cosd=sat_data['cosd'],
            tid=sat_data['tid'])
        satellites.append(s)

    with open('UEs.json', 'r') as fileUE:
        UEs_data = json.load(fileUE)

    # Process JSON data
    for UE_data in UEs_data:
        # Access satellite parameters
        ue = UE_template(
            ID=UE_data['index'],
            x=UE_data['x'],
            y=UE_data['y'])
        UEs.append(ue)

    with open('duration.txt', 'r') as file:
        duration_seconds = float(file.read().strip())

    return UEs, satellites, duration_seconds

UEs, Satellites, duration = load_scenario()

UEs_x = [ue.x for ue in UEs]
UEs_y = [ue.y for ue in UEs]

Sat_x = [satellite.x for satellite in Satellites]
Sat_y = [satellite.y for satellite in Satellites]
Sat_r = [satellite.r for satellite in Satellites]
Sat_tid = [satellite.tid for satellite in Satellites]

fig, ax = plt.subplots()
ax.scatter(UEs_x, UEs_y, color='green', alpha=0.3, s=2, marker='o')

for x, y, r, tid in zip(Sat_x, Sat_y, Sat_r, Sat_tid):
    if tid == 1:
        color = 'r'
    if tid == 2:
        color = 'b'
    if tid == 3:
        color = 'brown'
    if tid == 4:
        color = 'yellow'
    circle = plt.Circle((x, y), r, edgecolor=color, facecolor='none')
    ax.add_patch(circle)

# Adding labels and title
plt.xlabel('X coordinate (km)')
plt.ylabel('Y coordinate (km)')
plt.title('Scenario Start')
ax.set_aspect('equal', adjustable='box')
ax.grid(True)
plt.savefig('Scenario_start.png')  # Save the first plot
plt.close()  # Close the plot

for sat in Satellites:
    sat.update_location(duration)
    
Sat_x = [satellite.x for satellite in Satellites]
Sat_y = [satellite.y for satellite in Satellites]
Sat_r = [satellite.r for satellite in Satellites]
Sat_tid = [satellite.tid for satellite in Satellites]

fig, ax = plt.subplots()
ax.scatter(UEs_x, UEs_y, color='green', alpha=0.3, s=2, marker='o')

for x, y, r, tid in zip(Sat_x, Sat_y, Sat_r, Sat_tid):
    if tid == 1:
        color = 'r'
    if tid == 2:
        color = 'b'
    if tid == 3:
        color = 'brown'
    if tid == 4:
        color = 'yellow'
    circle = plt.Circle((x, y), r, edgecolor=color, facecolor='none')
    ax.add_patch(circle)

# Adding labels and title
plt.xlabel('X coordinate (km)')
plt.ylabel('Y coordinate (km)')
plt.title('Scenario End')
ax.set_aspect('equal', adjustable='box')
ax.grid(True)
plt.savefig('Scenario_end.png')  # Save the first plot
plt.close()  # Close the plot