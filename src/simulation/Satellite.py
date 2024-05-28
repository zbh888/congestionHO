import random

from Base import *
from Condition import *
from Config import *
from Counter import *
from Queue import *


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
                 max_access_opportunity,
                 max_access_slots,
                 oracle,
                 env):

        # Config Initialization
        Base.__init__(self,
                      identity=identity,
                      position_x=position_x,
                      position_y=position_y,
                      coverage_info=coverage_info,
                      env=env,
                      object_type="Satellite")

        self.coverage_r = coverage_r
        self.height = height
        self.velocity = velocity
        self.sind = sind
        self.cosd = cosd
        self.UEs = None
        self.access_Q = Queue(max_access_opportunity, max_access_slots)
        self.current_assigned_slot = None
        self.oracle = oracle

        # === source function ===
        # condition_record[ueid] stores the received conditions from candidates ([Sat_condition, ..., Sat_condition])
        self.condition_record = {}
        # candidates_record[ueid] stores candidates
        self.candidates_record = {}

        # === target function ===
        # takeover_condition_record[ueid] stores 
        self.takeover_condition_record = {}

        self.counter = counter(self.DURATION)

        # Running Process
        self.env.process(self.init())
        self.env.process(self.action_monitor())
        self.env.process(self.handle_messages())

    # ====== Satellite functions ======
    def action_monitor(self):
        while True:
            yield self.env.timeout(0.999999)
            self.current_assigned_slot = self.access_Q.shift()

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
                        "candidates": candidates,
                        "utility": data['utility'],
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
                candidates = data['candidates']
                utilities = data['utility']
                condition = self.prepare_condition(ueid, source_id, candidates, utilities)
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
                    best_targetid, delay = self.decide_best_target(ueid)
                    data = {
                        "task": RRC_RECONFIGURATION,
                        "conditions": self.condition_record[ueid],
                        "suggested_target": best_targetid,
                        "corresponding_delay": delay,
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
                assert (self.current_assigned_slot.include(ueid))
                assert (self.current_assigned_slot.time == self.env.now)
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

            # ================================================ Source
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
                self.access_Q.release_resource(ueid)

    # This is a source satellite function
    def decide_best_target(self, ueid):
        conditions = self.condition_record[ueid]
        if self.oracle is not None:
            targetid = self.oracle.query_next_satellite(ueid, self.identity)
        if self.oracle is not None and targetid != -1:
            # TODO not tested
            found = False
            for condition in conditions:
                if targetid == condition['satid']:
                    found = True
                    corresponding_delay = condition['access_delay']
            if not found:
                raise AssertionError("The UE did not receive condition from oracle arranged target satellite")
            return targetid, corresponding_delay
        # This clause if for non-oracle
        else:
            if SOURCE_ALG == SOURCE_ALG_RANDOM:
                selected_condition = random.choice(conditions)
            if SOURCE_ALG == SOURCE_ALG_EARLIEST:  # shortest delay
                min_delay = WINDOW_SIZE * 2
                condition_indices_with_shortest_delay = []
                for condition in conditions:
                    min_delay = min(min_delay, condition['access_delay'])
                for index, condition in enumerate(conditions):
                    if min_delay == condition['access_delay']:
                        condition_indices_with_shortest_delay.append(index)
                selected_condition = conditions[random.choice(condition_indices_with_shortest_delay)]
            if SOURCE_ALG == SOURCE_ALG_LONGEST:  # longest serving
                max_serving = -1
                condition_indices_with_max_serving = []
                for condition in conditions:
                    max_serving = max(max_serving, condition['ue_utility'])
                for index, condition in enumerate(conditions):
                    if max_serving == condition['ue_utility']:
                        condition_indices_with_max_serving.append(index)
                selected_condition = conditions[random.choice(condition_indices_with_max_serving)]
            targetid = selected_condition['satid']
            delay = selected_condition['access_delay']
            return targetid, delay

    # This is a candidate satellite function
    def decide_delay(self, ueid, sourceid, candidates, utilities):
        assert (self.access_Q.counter - self.access_Q.max_access_slots == self.env.now)
        available_slots = self.access_Q.available_slots()
        if CANDIDATE_ALG == CANDIDATE_ALG_EARLIEST:
            # greedy
            delay = min(available_slots) + 1
        if CANDIDATE_ALG == CANDIDATE_ALG_RANDOM:
            # random
            delay = random.choice(available_slots) + 1
        return delay

    def prepare_condition(self, ueid, sourceid, candidates, utilities):
        delay = self.decide_delay(ueid, sourceid, candidates, utilities)
        ue_utility = utilities[candidates.index(self.identity)]
        # TODO Should we consider the case when access time cannot happen with handover at the same time?
        if self.env.now + delay < self.DURATION:
            assert (self.coverage_info[ueid, self.identity, self.env.now + delay] == 1)
        condition = Sat_condition(access_delay=delay, ueid=ueid, satid=self.identity, sourceid=sourceid,
                                  ue_utility=ue_utility)
        self.access_Q.insert(ueid, delay)
        return condition
