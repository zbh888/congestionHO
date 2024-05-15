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


# This is a collection of Sat_condition
class UE_condition:
    def __init__(self, list_Sat_condition_json, ueid, creation_time):
        self.ueid = ueid
        self.conditions = {}
        for condition in list_Sat_condition_json:
            assert (self.ueid == condition["ueid"])
            self.conditions[condition['satid']] = (Sat_condition(access_delay=condition['access_delay'],
                                                 ueid=condition['ueid'],
                                                 satid=condition['satid'],
                                                 sourceid=condition['source']))
        self.creation_time = creation_time

    def available_access_conditions(self, time):
        available_conditons = []
        left_conditons = []
        for sat_id in self.conditions:
            condition = self.conditions[sat_id]
            if self.creation_time + condition.access_delay == time:
                available_conditons.append(condition)
            elif self.creation_time + condition.access_delay > time:
                left_conditons.append(condition)
        return available_conditons, left_conditons
