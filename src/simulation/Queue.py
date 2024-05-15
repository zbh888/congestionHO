from collections import deque

class slotQueue:
    def __init__(self, time, max_opportunity):
        self.time = time
        self.max_opportunity = max_opportunity


class Queue:
    def __init__(self, max_opportunity, max_access_slots):
        self.Q = deque()
        for i in range(max_access_slots):
            slotQ = slotQueue(i, max_opportunity)
