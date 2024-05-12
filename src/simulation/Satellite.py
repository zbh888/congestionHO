import random

from Base import *
from Condition import *
from Config import *
from Counter import *


class Satellite(Base):
    def __init__(self,
                 identity,
                 position_x,
                 position_y,
                 height,
                 coverage_r,
                 velocity,
                 sind,
                 cosd,
                 coverage_info,
                 env):

        # Config Initialization
        Base.__init__(self,
                      identity=identity,
                      position_x=position_x,
                      position_y=position_y,
                      coverage_info=coverage_info,
                      env=env,
                      object_type="Satellite")

        self.height = height
        self.velocity = velocity
        self.sind = sind
        self.cosd = cosd
        self.UEs = None

        # === source function ===
        # condition_record[ueid] stores the received conditions from candidates ([Sat_condition, ..., Sat_condition])
        self.condition_record = {}
        # condition_record[ueid] stores the condition that has been given (Sat_condition)
        self.candidates_record = {}

        # === target function ===
        # takeover_condition_record[ueid] stores 
        self.takeover_condition_record = {}

        self.counter = counter(self.DURATION)

        # Running Process
        self.env.process(self.init())
        self.env.process(self.handle_messages())

    # ====== Satellite functions ======
    def prepare_condition(self, ueid, sourceid):
        delay = random.randint(1, 20)
        # TODO Should we consider the case when access time cannot happen with handover at the same time?
        if self.env.now + delay < self.DURATION:
            assert (self.coverage_info[ueid, self.identity, self.env.now + delay] == 1)
        condition = Sat_condition(access_delay=delay, ueid=ueid, satid=self.identity, sourceid=sourceid)
        return condition

    def handle_messages(self):
        while True:
            msg = yield self.messageQ.get()
            data = json.loads(msg)
            assert (data['to'] == self.identity)
            now = self.env.now
            task = data['task']
            self.counter.increment(task, now)
            # ================================================ Source
            if task == MEASUREMENT_REPORT:
                ueid = data['from']
                candidates = data['candidates']
                assert (ueid not in self.condition_record)
                assert (ueid not in self.candidates_record)
                self.condition_record[ueid] = []
                self.candidates_record[ueid] = candidates
                for satid in candidates:
                    target_satellite = self.satellites[satid]
                    data = {
                        "task": HANDOVER_REQUEST,
                        "ueid": ueid,
                    }
                    self.send_message(
                        msg=data,
                        to=target_satellite
                    )
            # ================================================ Target + Candidate
            if task == HANDOVER_REQUEST:
                source_id = data['from']
                requested_satellite = self.satellites[source_id]
                ueid = data['ueid']
                condition = self.prepare_condition(ueid, source_id)
                data = {
                    "task": HANDOVER_RESPONSE,
                    "ueid": ueid,
                    "condition": condition.toJSON(),
                }
                self.send_message(
                    msg=data,
                    to=requested_satellite
                )
                self.takeover_condition_record[ueid] = condition
            # ================================================ Source
            if task == HANDOVER_RESPONSE:
                ueid = data["ueid"]
                condition = data["condition"]
                self.condition_record[ueid].append(condition)
                if len(self.condition_record[ueid]) == len(self.candidates_record[ueid]):
                    data = {
                        "task": RRC_RECONFIGURATION,
                        "conditions": self.condition_record[ueid],
                    }
                    UE_who_requested = self.UEs[ueid]
                    self.send_message(
                        msg=data,
                        to=UE_who_requested
                    )
            # ================================================ Target
            if task == RANDOM_ACCESS:
                # Response to UE
                ueid = data['from']
                takeover_condition = self.takeover_condition_record[ueid]
                UE_who_requested = self.UEs[ueid]
                data = {
                    "task": RANDOM_ACCESS_RESPONSE,
                }
                self.send_message(
                    msg=data,
                    to=UE_who_requested
                )

                # Response to source Satellite
                sourceid = takeover_condition.sourceid
                source = self.satellites[sourceid]
                data = {
                    "task": HANDOVER_SUCCESS,
                    "ueid": ueid
                }
                self.send_message(
                    msg=data,
                    to=source
                )
                # upon receiving random access, the target delete the condition record and take over UE
                del self.takeover_condition_record[ueid]
            # ================================================ Source
            if task == HANDOVER_SUCCESS:
                # send SN status transfer
                target_id = data['from']
                target = self.satellites[target_id]
                ueid = data['ueid']
                data = {
                    "task": SN_STATUS_TRANSFER,
                    "ueid": ueid
                }
                self.send_message(
                    msg=data,
                    to=target
                )

                # cancel other candidates
                for candidateid in self.candidates_record[ueid]:
                    if candidateid != target_id:
                        candidate = self.satellites[candidateid]
                        data = {
                            "task": HANDOVER_CANCEL,
                            "ueid": ueid
                        }
                        self.send_message(
                            msg=data,
                            to=candidate
                        )
                # Upon receiving handover success, the source remove the UE's record
                del self.candidates_record[ueid]
                del self.condition_record[ueid]

            # ================================================ 
            if task == RRC_RECONFIGURATION_COMPLETE:
                # no logic needs to be handled here
                assert (True)
                # ================================================ Target
            if task == SN_STATUS_TRANSFER:
                assert (True)
            # ================================================ Candidate
            if task == HANDOVER_CANCEL:
                ueid = data['ueid']
                # upon receving handover cancel, the candidate remove the UE's record
                del self.takeover_condition_record[ueid]
