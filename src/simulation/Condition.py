class Sat_condition:
    def __init__(self, access_delay, ueid, satid, sourceid, ue_utility):
        self.access_delay = access_delay
        self.ueid = ueid
        self.satid = satid
        self.sourceid = sourceid
        self.ue_utility = ue_utility

    def toJSON(self):
        return {
            "access_delay": self.access_delay,
            "ueid": self.ueid,
            "source": self.sourceid,
            "satid": self.satid,
            "ue_utility": self.ue_utility
        }

class UE_condition:
    def __init__(self, targetid, delay, now):
        self.targetid = targetid
        self.delay = delay
        self.access_time = delay + now

