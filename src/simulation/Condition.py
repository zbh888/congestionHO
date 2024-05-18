class Sat_condition:
    def __init__(self, access_delay, ueid, satid, sourceid):
        self.access_delay = access_delay
        self.ueid = ueid
        self.satid = satid
        self.sourceid = sourceid

    def toJSON(self):
        return {
            "access_delay": self.access_delay,
            "ueid": self.ueid,
            "source": self.sourceid,
            "satid": self.satid
        }

class UE_condition:
    def __init__(self, targetid, delay, now):
        self.targetid = targetid
        self.delay = delay
        self.access_time = delay + now

