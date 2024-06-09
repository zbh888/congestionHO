from collections import deque
import numpy as np


class slotQueue:
    def __init__(self, time, max_opportunity):
        # The time index indicate that this slot is for that time
        self.time = time
        self.max_opportunity = max_opportunity
        self.UE_list = deque()

    def length(self):
        return len(self.UE_list)

    def insert(self, ueid):
        if self.length() == self.max_opportunity:
            raise AssertionError("It is already full")
        self.UE_list.append(ueid)

    def include(self, ueid):
        return (ueid in self.UE_list)

    def delete(self, ueid):
        self.UE_list.remove(ueid)


class Queue:
    def __init__(self, max_opportunity, max_access_slots):
        self.access_issue_time_delay = {}
        self.counter = max_access_slots
        self.max_access_slots = max_access_slots
        self.max_opportunity = max_opportunity
        self.slots_status = deque()
        self.Q = deque()
        for i in range(max_access_slots):
            slotQ = slotQueue(i + 1, max_opportunity)
            self.Q.append(slotQ)
            self.slots_status.append(self.max_opportunity)

        self.reserved_number = 0

    def calulate_reservation_rate(self):
        return self.reserved_number / (self.max_opportunity * self.max_access_slots)

    def shift(self):
        # Note this counter is not simulation time
        # simulation time = self.counter - max_access_slots
        self.counter += 1
        slotQ = slotQueue(self.counter, self.max_opportunity)
        self.slots_status.popleft()
        self.slots_status.append(self.max_opportunity)
        #  rate = self.calulate_reservation_rate()
        Q = self.Q.popleft()
        temp = self.reserved_number
        self.reserved_number -= Q.length()
        self.Q.append(slotQ)
        return Q, temp

    def insert(self, ueid, delay):
        self.access_issue_time_delay[ueid] = (self.counter - self.max_access_slots, delay)
        self.Q[delay - 1].insert(ueid)
        self.slots_status[delay - 1] -= 1
        self.reserved_number += 1  # just for reservation count

    def available_slots(self):
        return np.array([True if value > 0 else False for value in self.slots_status])

    def release_resource(self, ueid):
        issue_time, expected_time = self.return_expected_issue_access_time(ueid)
        slot = expected_time - self.counter + self.max_access_slots - 1
        if slot >= 0:
            self.Q[slot].delete(ueid)
            self.slots_status[slot] += 1
            self.reserved_number -= 1  # just for reservation count

    def return_expected_issue_access_time(self, ueid):
        return self.access_issue_time_delay[ueid][0], self.access_issue_time_delay[ueid][0] + \
               self.access_issue_time_delay[ueid][1]
