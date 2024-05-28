from collections import deque


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

    def shift(self):
        # Note this counter is not simulation time
        # simulation time = self.counter - max_access_slots
        self.counter += 1
        slotQ = slotQueue(self.counter, self.max_opportunity)
        self.slots_status.popleft()
        self.slots_status.append(self.max_opportunity)
        Q = self.Q.popleft()
        self.Q.append(slotQ)
        return Q

    def insert(self, ueid, delay):
        self.access_issue_time_delay[ueid] = (self.counter - self.max_access_slots, delay)
        self.Q[delay - 1].insert(ueid)
        self.slots_status[delay - 1] -= 1

    def available_slots(self):
        return [index for index, value in enumerate(self.slots_status) if value > 0]

    def release_resource(self, ueid):
        expected_time = self.access_issue_time_delay[ueid][0] + self.access_issue_time_delay[ueid][1]
        slot = expected_time - self.counter + self.max_access_slots - 1
        if slot >= 0:
            self.Q[slot].delete(ueid)
            self.slots_status[slot] += 1
