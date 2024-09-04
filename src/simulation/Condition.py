class Sat_condition:
    def __init__(self, access_delay, ueid, satid, sourceid, ue_utility, future_potential_load, future_real_load, issue_time):
        self.access_delay = access_delay
        self.ueid = ueid
        self.satid = satid
        self.sourceid = sourceid
        self.ue_utility = ue_utility
        self.future_potential_real_load = (future_potential_load, future_real_load)
        self.issue_time = issue_time

    def toJSON(self):
        return {
            "access_delay": self.access_delay,
            "ueid": self.ueid,
            "source": self.sourceid,
            "satid": self.satid,
            "ue_utility": self.ue_utility,
            'future_potential_real_load': self.future_potential_real_load,
            'issue_time': self.issue_time
        }


class UE_condition:
    def __init__(self, targetid, delay, now):
        self.targetid = targetid
        self.delay = delay
        self.access_time = delay + now
